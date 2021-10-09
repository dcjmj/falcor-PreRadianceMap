#include "Utils/Math/MathConstants.slangh"
#include "Utils/HostDeviceShared.slangh"
#include "Experimental/Scene/Lights/EmissiveLightSamplerType.slangh"

import Scene.Scene;
import Scene.Raytracing;
import Utils.Math.MathHelpers;
import Utils.Helpers;
import Utils.Sampling.SampleGenerator;
import Scene.Shading;
import Experimental.Scene.Material.TexLODHelpers;
import Experimental.Scene.Material.TexLODTypes;
import Scene.HitInfo;

import Experimental.Scene.Lights.EnvMapSampler;
import Experimental.Scene.Lights.EmissiveLightSampler;  
import Experimental.Scene.Material.MaterialShading;
import Experimental.Scene.Lights.LightHelpers;


static const uint kSamplesPerPixel = 400;
static const uint kLightSamplesPerVertex = 5;
static const uint kMaxBounces = 5;
static const uint kMaxNonSpecularBounces = 2;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const bool kUseEmissiveSampler = true;
static const bool kUseBRDFSampling = true;
static const bool kUseMIS = true;
static const bool kUseRussianRoulette = true;
static const bool kComputeIndirect = true;

// Define max supported path length and number of shadow rays per vertex.
// These limits depend on the bit layout for the packed path flags.
static const uint kMaxPathFlagsBits = 16;
static const uint kMaxPathLengthBits = 8;
static const uint kMaxPathLength = (1 << kMaxPathLengthBits) - 1;
static const uint kMaxLightSamplesPerVertex = 25;
static const uint kMaxRejectedHits = 16; // Maximum number of rejected hits along a path. The path is terminated if the limit is reached to avoid getting stuck in pathological cases.

// Define ray indices.
static const uint32_t kRayTypeScatter = 0;
static const uint32_t kRayTypeShadow = 1;

static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

// Logic for determining if we need to trace a scatter ray from the last path vertex.
// This is needed to account for direct illumination when NEE with MIS is enabled, or NEE is disabled.
// Note that these decisions are all static, no dynamic code is generated.
static const bool kTraceScatterRayFromLastPathVertex = true;

static float specularRoughnessThreshold = 0.25f;
static float probabilityAbsorption = 0.2f;
static int     clampDirect = true;            ///< Clamp the radiance for direct illumination samples to 'thresholdDirect' to reduce fireflys.
static int     clampIndirect = true;          ///< Clamp the radiance for indirect illumination samples to 'thresholdIndirect' to reduce fireflys.
static float   thresholdDirect = 10.f;
static float   thresholdIndirect = 10.f;

/** A light sample for any of the scene lights (analytic, envmap, and emissive).
    The Li field is nonzero only if the sample is valid (no need to check the pdf).
*/
struct SceneLightSample
{
    // Light sample
    float3  dir;            ///< Direction from the shading point to the light sample in world space (normalized). This is used for BRDF evaluation.
    float   distance;       ///< Distance from the shading point to the light sample.
    float3  Li;             ///< Incident radiance at the shading point (unshadowed). Note: Already divided by the pdf and multiplied by MIS weight.
    float   pdf;            ///< Probability density function with respect to solid angle (pdf == 0 for invalid samples).

    // Shadow ray parameters
    float3  rayDir;         ///< Ray direction for visibility evaluation (normalized). This may differ from 'dir' due to ray offsets.
    float   rayDistance;    ///< Ray distance for visibility evaluation. This may differ from 'distance' due to ray offsets.
};

// TODO: Move enum to host/device shared header so that we can add validation.
/** The path flags are stored in `kMaxPathFlagsBits` bits.
    If all ray bits are zero the path is inactive/terminated.
*/
enum class PathFlags
{
    // Bits 0-7 path flags (8 bits)
    scatterRay      = 0x0001,   ///< Scatter ray is active.
    scatterHit      = 0x0002,   ///< Result of the scatter ray (0 = miss, 1 = hit).
    transmission    = 0x0004,   ///< Scatter ray went through a transmission event.
    specular        = 0x0008,   ///< Scatter ray went through a specular event.
    delta           = 0x0010,   ///< Scatter ray went through a delta event.
    insideVolume    = 0x0020,   ///< Path vertex is inside a volume.
    // Bits 6-7 reserved

    // Bits 8-15 shadow ray active (8 bits)
    shadowRay       = 0x0100,   ///< Shadow ray is active (0 = inactive/occluded, 1 = active/visible)

    anyShadowRays   = 0xff00,   ///< Bit mask to test if there are any active shadow rays.
    anyRays         = 0xff01,   ///< Bit mask to test if there are any active rays (scatter and/or shadow).
};

/** Working data for the path tracer.

    Note that the shadow ray data is handled separately to avoid having a very
    large live state when there are multiple light samples per vertex.
*/
struct PathData
{
    uint    flags;              ///< Flags indicating the current status. This can be multiple PathFlags flags OR'ed together.
    uint    length;             ///< Path length (0 at origin, 1 at first secondary hit, etc.).
    uint    nonSpecularBounces; ///< Number of non-specular bounces along the path.

    // Scatter ray
    float3  origin;             ///< Origin of the shadow/scatter rays. Note: This field is only loaded if MIS is enabled.
    float3  transmissionOrigin; ///< Origin of the scatter ray after a transmission event. Note: This is write only.
    float3  dir;                ///< Scatter ray normalized direction.
    float3  thp;                ///< Path throughput.
    float   pdf;                ///< Pdf for generating the scatter ray. Note: This field is only loaded/stored if MIS is enabled.
    float3  normal;             ///< Shading normal at the scatter ray origin. Note: This field is only loaded/stored if MIS is enabled.
    // HitInfo hit;                ///< Hit information for the scatter ray. This is populated by the tracing pass. Only valid if the flag 'scatterHit' is set.

    // Common data
    float3  L;                  ///< Accumulated path contribution.

    // InteriorList interiorList;  ///< Interior list. Keeping track of a stack of materials with medium properties.

    SampleGenerator sg;         ///< Sample generator state. Note: This is only valid when path.length < kMaxBounces.

    // RayFootprint rayFootprint;  ///< Ray footprint for TexLOD. Actual data depends on the footprint mode.

    // Utility functions

    bool isTerminated() { return !hasFlag(PathFlags::anyRays); }
    bool isTransmission() { return hasFlag(PathFlags::transmission); }
    bool isSpecular() { return hasFlag(PathFlags::specular); }
    bool isDelta() { return hasFlag(PathFlags::delta); }
    bool isInsideVolume() { return hasFlag(PathFlags::insideVolume); }

    // Check if the scatter event is samplable by the light sampling technique.
    bool isLightSamplable() { return !(isDelta() || isTransmission()); }

    bool hasRays() { return hasFlag(PathFlags::anyRays); }
    bool hasScatterRay() { return hasFlag(PathFlags::scatterRay); }
    bool hasShadowRay(uint i) { return (flags & (uint(PathFlags::shadowRay) << i)) != 0; }
    bool isShadowRayOccluded(uint i) { return (flags & ((uint)PathFlags::shadowRay << i)) == 0; }

    [mutating] void clearScatterRay() { flags &= ~(uint(PathFlags::scatterRay) | uint(PathFlags::scatterHit)); }
    [mutating] void clearShadowRay(uint i) { flags &= ~(uint(PathFlags::shadowRay) << i); }
    [mutating] void clearShadowRays() { flags &= ~(uint(PathFlags::anyShadowRays)); }

    [mutating] void clearEventFlags() { flags &= ~(uint(PathFlags::transmission) | uint(PathFlags::specular) | uint(PathFlags::delta)); }
    [mutating] void setTransmission(bool value = true) { setFlag(PathFlags::transmission, value); }
    [mutating] void setSpecular(bool value = true) { setFlag(PathFlags::specular, value); }
    [mutating] void setDelta(bool value = true) { setFlag(PathFlags::delta, value); }
    [mutating] void setInsideVolume(bool value = true) { setFlag(PathFlags::insideVolume, value); }

    bool hasFlag(PathFlags flag) { return (flags & uint(flag)) != 0; }
    [mutating] void setFlag(PathFlags flag, bool value = true)
    {
        if (value) flags |= uint(flag);
        else flags &= ~uint(flag);
    }
};

/** Evaluates the background in a particular direction.
    This function should be called for screen-space samples that missed the scene.
    The result should not be used as lighting, use evalDistantLight() instead.
    \param[in] dir World-space direction (unnormalized).
    \return Color (rgb).
*/
float3 evalBackground(float3 dir)
{
    return kUseEnvBackground ? gScene.envMap.eval(dir) : kDefaultBackgroundColor;
}

/** uniform sample in a hemisphere
when u is (0.5,0.5),sample the normal dir
*/
float3 sample_uniform_hemisphere(float2 u, out float pdf)
{
    float2 d = sample_disk_concentric(u);
    float z = sqrt(max(0.f, 1.f - dot(d, d)));
    pdf = z * M_1_PI;
    return float3(d, z);
    /**
    float rho = u.x * M_PI / 2.0;
    float theta =  u.y * 2.0 * M_PI;
    float x = 1.f * sin(rho) * cos(theta);
    float y = 1.f * sin(rho) * sin(theta);
    float z = 1.f * cos(rho);
    pdf = M_1_PI / 2;
    return float3(x, y, z);
    */
}

/** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
    This version assumes one sample is taken from each of the sampling strategies.
    \param[in] params PathTracer parameters.
    \param[in] p0 Pdf for the first sampling strategy.
    \param[in] p1 Pdf for the second sampling strategy.
    \return Weight for the contribution from the first strategy (p0).
*/
float evalMIS(float p0, float p1)
{
    return p0 * p0 / (p0 * p0 + p1 * p1);
}

/** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
    \param[in] params PathTracer parameters.
    \param[in] n0 Number of samples taken from the first sampling strategy.
    \param[in] p0 Pdf for the first sampling strategy.
    \param[in] n1 Number of samples taken from the second sampling strategy.
    \param[in] p1 Pdf for the second sampling strategy.
    \return Weight for the contribution from the first strategy (p0).
*/
float evalMIS(float n0, float p0, float n1, float p1)
{
    float q0 = (n0 * p0) * (n0 * p0);
    float q1 = (n1 * p1) * (n1 * p1);
    return q0 / (q0 + q1);
}

/** Returns the probability of selecting env map light sampling.
*/
float getEnvLightSelectionPdf()
{
    float p0 = kUseEnvLight ? 1.f : 0.f;
    float p1 = kUseAnalyticLights ? 1.f : 0.f;
    float p2 = kUseEmissiveLights ? 1.f : 0.f;
    return p0 / (p0 + p1 + p2);
}

/** Returns the probability of selecting emissive light sampling.
*/
float getEmissiveLightSelectionPdf()
{
    float p0 = kUseEnvLight ? 1.f : 0.f;
    float p1 = kUseAnalyticLights ? 1.f : 0.f;
    float p2 = kUseEmissiveLights ? 1.f : 0.f;
    return p2 / (p0 + p1 + p2);
}

/** Evaluates the probability density function for the BRDF sampling strategy used for the scatter ray.
    \param[in] sd Describes the shading point.
    \param[in] dir The normalized incident direction for which to evaluate the pdf.
    \return Probability density with respect to solid angle from the shading point.
*/
float evalPdfScatter(const ShadingData sd, const float3 dir)
{
    if (kUseBRDFSampling)
    {
        return evalPdfBSDF(sd, dir);
    }
    else // Cosine-weighted sampling
    {
        // Check the dot products. The sampling probability for back-facing directions is zero.
        float NdotL = dot(sd.N, dir);
        if (min(sd.NdotV, NdotL) < kMinCosTheta) return 0.f;

        return NdotL * M_1_PI; // pdf = cos(theta) / pi
    }
}

/** Samples a light source in the scene.
    This function first stochastically selects a type of light source to sample,
    and then calls that the sampling function for the chosen light type.
    \param[in] params PathTracer parameters.
    \param[in] envMapSampler Environment map sampler.
    \param[in] emissiveSampler Emissive light sampler.
    \param[in] sd Shading data.
    \param[in] rayOrigin Ray origin for the shadow ray.
    \param[in] numSamples Total number of light samples that will be taken. This is needed for the MIS weighting.
    \param[in,out] sg SampleGenerator object.
    \param[out] ls Generated light sample. Only valid if true is returned.
    \return True if a sample was generated, false otherwise.
*/
bool sampleSceneLights(const EnvMapSampler envMapSampler, const EmissiveLightSampler emissiveSampler, const ShadingData sd, const float3 rayOrigin, const uint numSamples, inout SampleGenerator sg, out SceneLightSample ls)
{
    // Set relative probabilities of the different sampling techniques.
    // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
    // TODO: Refactor the selection pdf code the helpers above for evaluating the individual probabilities.
    float p[3];
    p[0] = kUseEnvLight ? 1.f : 0.f;
    p[1] = kUseAnalyticLights ? 1.f : 0.f;
    p[2] = kUseEmissiveLights ? 1.f : 0.f;

    // Normalize probabilities. Early out if zero.
    float sum = p[0] + p[1] + p[2];
    if (sum == 0.f) return false;

    float invSum = 1.f / sum;
    p[0] *= invSum;
    p[1] *= invSum;
    p[2] *= invSum;

    // Sample based on uniform random number. Rescale u to [0,1) afterwards.
    float invPdf = 0.f;
    float u = sampleNext1D(sg);

    // We use explicit checks for which light types are enabled so that the compiler
    // can remove the unused code. It won't otherwise since u is unknown at compile time.
    if (kUseEnvLight)
    {
        if (u < p[0])
        {
            float selectionPdf = p[0];

            // Sample environment map.
            EnvMapSample lightSample;
            envMapSampler.sample(sampleNext2D(sg), lightSample);

            // Reject sample if lower hemisphere.
            if (dot(sd.N, lightSample.dir) < kMinCosTheta) return false;

            // Evaluate MIS with BRDF sampling as the other sampling strategy.
            float pdf = selectionPdf * lightSample.pdf;
            float misWeight = 1.f;
            if (kUseMIS && pdf > 0.f)
            {
                float brdfPdf = evalPdfScatter(sd, lightSample.dir);
                misWeight = evalMIS(numSamples, pdf, 1, brdfPdf);
            }

            // Setup returned sample.
            ls.rayDir = ls.dir = lightSample.dir;
            ls.rayDistance = ls.distance = kRayTMax;
            ls.pdf = pdf;
            ls.Li = pdf > 0.f ? lightSample.Le * misWeight / (pdf * numSamples) : float3(0);
            //下面这句debug用
            //ls.Li = lightSample.Le;

            return true;
        }
        u -= p[0];
    }

    if (kUseAnalyticLights)
    {
        if (u < p[1])
        {
            // Sample analytic light source selected uniformly from the light list.
            // TODO: Sample based on estimated contributions as pdf.
            u /= p[1]; // Rescale to [0,1)
            uint lightCount = gScene.getLightCount();
            uint lightIndex = min(uint(u * lightCount), lightCount - 1);
            float selectionPdf = p[1] / lightCount; // TODO: Precompute 1.f/lightCount in cbuffer

            // Sample local light source.
            AnalyticLightSample lightSample;
            bool valid = sampleLight(rayOrigin, gScene.getLight(lightIndex), sg, lightSample);

            // Reject sample if lower hemisphere.
            if (!valid || dot(sd.N, lightSample.dir) < kMinCosTheta) return false;

            // Setup returned sample.
            // Analytic lights do not currently have a geometric representation in the scene.
            // Do not worry about adjusting the ray to avoid self-intersections at the light.
            ls.rayDir = ls.dir = lightSample.dir;
            ls.rayDistance = ls.distance = lightSample.distance;
            ls.pdf = selectionPdf * lightSample.pdf;
            ls.Li = lightSample.Li / (selectionPdf * numSamples);
            //下面这句debug用
            //ls.Li = lightSample.Li;
            return true;
        }
        u -= p[1];
    }

    if (kUseEmissiveLights)
    {
        //if (u < p[2]) // Always true
        {
            float selectionPdf = p[2];

            // Sample emissive lights.computeRayOrigin
            TriangleLightSample lightSample;
            bool valid = emissiveSampler.sampleLight(rayOrigin, sd.N, sg, lightSample);

            // Reject sample if lower hemisphere.
            if (!valid || dot(sd.N, lightSample.dir) < kMinCosTheta) return false;

            // Evaluate MIS with BRDF sampling as the other sampling strategy.
            float pdf = selectionPdf * lightSample.pdf;
            float misWeight = 1.f;
            if (kUseMIS && pdf > 0.f)
            {
                float brdfPdf = evalPdfScatter(sd, lightSample.dir);
                misWeight = evalMIS(numSamples, pdf, 1, brdfPdf);
            }

            // Compute offset light sample position to reduce self-intersections at the light.
            // We compute the shadow ray parameters based on the offset position.
            float3 offsetPos = computeRayOrigin(lightSample.posW, lightSample.normalW);
            float3 toLight = offsetPos - rayOrigin;
            ls.rayDistance = length(toLight);
            ls.rayDir = normalize(toLight);

            // Setup returned sample.
            ls.dir = lightSample.dir;
            ls.distance = lightSample.distance;
            ls.pdf = pdf;
            ls.Li = pdf > 0.f ? lightSample.Le * misWeight / (pdf * numSamples) : float3(0);
            //下面这句debug用
            //ls.Li = lightSample.Le;

            return true;
        }
    }

    return false;
}

/** Generates a shadow ray for sampling the light sources.
    This should be called before generateScatterRay() as the latter updates the path throughput.
    The function assumes path.origin is the ray origin for the shadow ray, and that the PathData
    flags for the shadow ray have already been cleared.
    \param[in] params Path tracer parameters.
    \param[in] envMapSampler Environment map sampler.
    \param[in] emissiveSampler Emissive light sampler.
    \param[in] sd Shading data.
    \param[in] i The sample index in the range [0, kLightSamplesPerVertex).
    \param[in,out] path Path data. The path flags will be updated to enable the i:th shadow ray if a sample was generated.
    \param[in,out] sg Sample generator.
    \param[in,out] shadowRay Shadow ray parameters and unoccluded contribution for the generated sample.
    \return True if a sample was generated, false otherwise.
*/
bool generateShadowRay(const EnvMapSampler envMapSampler, const EmissiveLightSampler emissiveSampler, const ShadingData sd, const uint i, inout PathData path, inout SampleGenerator sg, inout ShadowRay shadowRay)
{
    // Sample the scene lights.
    SceneLightSample ls;
    bool valid = sampleSceneLights(envMapSampler, emissiveSampler, sd, path.origin, kLightSamplesPerVertex, sg, ls);

    if (valid && any(ls.Li > 0.f))
    {
        float3 Lr = evalBSDFCosine(sd, ls.dir) * ls.Li * path.thp;
        if (any(Lr > 0.f))
        {
            // The sample is valid and has a non-zero contribution.
            // Store ray parameters and unoccluded radiance, weighted by path throughput.
            path.flags |= (uint(PathFlags::shadowRay) << i);
            shadowRay.rayParams = float4(ls.rayDir, ls.rayDistance);
            shadowRay.Lr = Lr;
            return true;
        }
    }
    return false;
}

/** Generates a scatter ray or terminates the path.
    The function uses BSDF sampling to generate ray parameters for the scatter ray.
    The function assumes path.origin is the ray origin for the new ray.
    \param[in] params Path tracer parameters.
    \param[in] sd Shading data.
    \param[in,out] path Path data.
    \param[in,out] sg Sample generator.
    \return True if a sample was generated and path should continue, false otherwise.
*/
void generateScatterRay(const ShadingData sd, const float3 rayOrigin, inout ScatterRayData rayData)
{
     // Generate next path segment.
    bool valid = false;
    rayData.origin = rayOrigin;
    if (kUseBRDFSampling)
    {
        // Default path that uses BSDF importance sampling.
        BSDFSample result;
        valid = sampleBSDF(sd, rayData.sg, result);
        if(valid)
        {
            rayData.direction = result.wi;
            rayData.thp *= result.weight;

            // Note that the pdf can be costly, so we store it in the payload if we need it (MIS enabled),
            // and rely on dead code elimination to remove the computations otherwise.
            rayData.pdf = result.pdf;
        }
    }
    else
    {
        // Fallback path that uses cosine-weighted hemisphere sampling.
        float pdf;
        float3 dir = sampleHemisphereCosine(sd, sampleNext2D(rayData.sg), pdf); // pdf = cos(theta) / pi

        // Check that L and V are in the positive hemisphere, reset pdf to zero otherwise.
        // This is necessary for consistency with BRDF sampling.
        float NdotL = dot(sd.N, dir);
        if (min(sd.NdotV, NdotL) < kMinCosTheta) pdf = 0.f;

        rayData.direction = dir;
        rayData.thp *= evalBSDF(sd, dir) * M_PI; // dot(N,L) / pdf = pi
        rayData.pdf *= kUseMIS ? pdf : 0.f;

        valid = true;
    }

    // Pass on the shading normal. Only store it if needed (MIS enabled).
    if (kUseMIS) rayData.normal = sd.N;

    // Terminate if ray throughput is zero.
    // This may happen if the sample is invalid, and consequently the BRDF sampling function returns thp == 0.0.
    if (all(rayData.thp == 0.f)) rayData.terminated = true;
    if (!valid) rayData.terminated = true;
}

cbuffer CB{
    uint guSpp;
    uint gvSpp;
    uint gVertexCount;
};

//TODO delete the pathTracerParams
struct PathTracerData
{
    // PathTracerParams        params;             ///< PathTracer shared parameters.
    EnvMapSampler           envMapSampler;      ///< Environment map sampler.
    EmissiveLightSampler    emissiveSampler;    ///< Emissive light sampler.
};

shared Buffer<uint2> gVertexToMesh;
shared RWStructuredBuffer<float> gTcoeffsR;
shared RWStructuredBuffer<float> gTcoeffsG;
shared RWStructuredBuffer<float> gTcoeffsB;

ParameterBlock<PathTracerData> gData;

//output channels
shared RWTexture2D<float4> gOutGoingRadiance;
shared RWTexture2D<float4> gOutGoingDistance;
shared RWTexture2D<float4> gOutGoingNormal;


struct ShadowRay
{
    float4 rayParams;           ///< Shadow ray normalized direction (xyz) and distance (w).
    float3 Lr;                  ///< Unoccluded contribution from the shadow ray (xyz).
    float _pad;
};

struct ShadowRayData{
    bool visible;
};

struct Trianglehit
{
    float3  posW;               ///< Sampled point on the triangle in world space.
    float3  normalW;            ///< Face normal of the triangle in world space, flipped for back-facing hits.
    float   triangleArea;       ///< Area of the triangle in world space.
    uint    meshInstanceID;     ///< Mesh instance ID (= global hit ID).
    uint    primitiveIndex;     ///< Primitive index in mesh.
};

/** Unpacked payload for scatter ray.   
*/
struct ScatterRayData
{
    bool terminated;        ///< Set to true when path is terminated.
    uint pathLength;        ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.).

    float3 Le;              ///< Emitted radiance at the last path vertex, weighted by path throughput and MIS weight.
    float3 Lr;              ///< Reflected radiance from the sampled light at the last path vertex, weighted by path throughput, BRDF, and MIS weight.

    float3 origin;          ///< Next path segment origin (unless path is terminated).
    float3 direction;       ///< Next path segment direction (unless path is terminated).

    float3 thp;             ///< Current path throughput. This is updated at each path vertex.

    float3 normal;          ///< Shading normal at the ray origin. Only used if MIS is enabled.
    float pdf;              ///< Probability density function with respect to solid angle at the last path vertex. Only used if MIS is enabled.

    float3 hitnormal;
    float3 distance;
    float4 shadowRay;       ///< Shadow ray direction (xyz) and distance (w). Only used if Lr > 0.

    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).

    /** Create ray payload with default parameters.
    */
    static ScatterRayData create(SampleGenerator sg)
    {
        ScatterRayData d;
        d.terminated = false;
        d.pathLength = 0;
        d.Le = float3(0, 0, 0);
        d.Lr = float3(0, 0, 0);
        d.origin = float3(0, 0, 0);
        d.direction = float3(0, 0, 0);
        d.thp = float3(1, 1, 1);
        d.normal = float3(0, 0, 0);
        d.pdf = 0.f;
        d.shadowRay = float4(0, 0, 0, 0);
        d.sg = sg;
        d.hitnormal = float3(0, 0, 0);
        d.distance = float3(0, 0, 0);

        return d;
    }
};

/** Traces a shadow ray towards a light source.
    \param[in] origin Ray origin for the shadow ray.
    \param[in] dir Direction from ray origin towards the light source (normalized).
    \param[in] distance Distance to the light source.
    \param[in] valid True if ray should be traced, false for dummy rays.
    \return True if light is visible, false otherwise.
*/
bool traceShadowRay(float3 origin, float3 dir, float distance, bool valid = true)
{
    // Setup ray descriptor.
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = valid ? distance : 0.f; // Set tmax = 0.0 for dummy rays.

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before
    uint rayFlags = RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    TraceRay(gRtScene, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeShadow /* hitIdx */, hitProgramCount, kRayTypeShadow /* missIdx */, ray, rayData);

    // We let TraceRay() execute even when !valid in order to run the miss shader.
    // This is faster on current drivers thanks to reduced divergence.
    if (!valid) return false;


    return rayData.visible;
}

/** Traces a scatter ray based on ray parameters stored in the ray payload.
    \param[in,out] rayData Describes the ray parameters. The struct is modified based on the result.
*/
void traceScatterRay(inout ScatterRayData rayData)
{
    // Setup ray based on params passed via payload.
    RayDesc ray;
    ray.Origin = rayData.origin;
    ray.Direction = rayData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    // Pack ray payload.
    uint rayFlags = 0; // TODO: Set cull mode from the app
    TraceRay(gRtScene, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatter /* hitIdx */, hitProgramCount, kRayTypeScatter /* missIdx */, ray, rayData);

    // Unpack ray payload.
}

/** Handles the case when a path misses the scene.
    Depending on the configuration, we add the env map contribution here using MIS.
    \param[in] pt PathTracer data.
    \param[in,out] rayData Ray payload.
*/
void handleMiss(const PathTracerData pt, inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the path as terminated.
    rayData.terminated = true;

    // If we have an environment map and MIS is enable, add the weighted contribution here.
    if (kUseEnvLight && kUseMIS)
    {
        const float3 dir = rayData.direction;
        float lightPdf = pt.envMapSampler.evalPdf(dir) * getEnvLightSelectionPdf();

        // Compute MIS weighted contribution from the environment map.
        float misWeight = evalMIS(1, rayData.pdf, 1, lightPdf);

        float3 Le = pt.envMapSampler.eval(dir);

        // Accumulate emitted radiance weighted by path throughput.0
        //rayData.Le = rayData.direction;
        rayData.Le += rayData.thp * Le  * misWeight;

    }
    //rayData.Le = float3(0, 0, 0);
}

/** Handles the case when a path hits a surface.
    Depending on the configuration, we add the emissive contribution here using MIS.
    Then direct illumination is sampled and the next     segment of the path generated.
    \param[in] pt PathTracer data.
    \param[in] sd Shading data.
    \param[in] hit Triangle hit data (for EmissiveLightSampler).
    \param[in] computeIndirect True if indirect illumination is enabled.
    \param[in,out] rayData Ray payload.
*/
void handleHit(const PathTracerData pt, const ShadingData sd, const Trianglehit hit, const bool computeIndirect, inout ScatterRayData rayData)
{
    // Statically determine if we need to compute the emissive based on the current configuration.
    // It's only needed if emissive is enabled, and its full contribution hasn't been sampled elsewhere.
    const bool computeEmissive = kUseEmissiveLights && (!kUseEmissiveSampler || (kUseEmissiveSampler && kUseMIS));
    if (computeEmissive && any(sd.emissive > 0.f))
    {
        float misWeight = 1.f;
        if (kUseEmissiveSampler && kUseMIS)
        {
            // If emissive light sampling and MIS is enabled, we've already sampled emissive lights using NEE.
            // We need to evaluate the MIS weight here to account for the remaining contribution.

            // Evaluate PDF at the hit, had it been generated with light sampling.
            // Note that there is a minor discrepancy compared to BRDF sampling at the previous path vertex:
            // the normal passed through the ray payload is lossily compressed to reduce the ray payload size.
            TriangleHit Thit;
            Thit.triangleIndex = gScene.lightCollection.getTriangleIndex(hit.meshInstanceID, hit.primitiveIndex);;
            Thit.posW = hit.posW;
            Thit.normalW = hit.normalW;

            float lightPdf = pt.emissiveSampler.evalPdf(rayData.origin, rayData.normal, Thit) * getEmissiveLightSelectionPdf();

            // Compute MIS weight by combining this with BRDF sampling.
            // Note we can assume rayData.pdf > 0.f since we shouldn't have got here otherwise.
            misWeight = evalMIS(rayData.pdf, lightPdf);
        }

        // Return emitted radiance weighted by path throughput and MIS weight.
        rayData.Le = rayData.thp * sd.emissive * misWeight;
    }

    // Unless path ends here, sample light and generate next path segment.
    // TODO: We should try running a different shader for the last path segment to see if that's faster than branching out here.
    if (computeIndirect && rayData.pathLength < kMaxBounces)
    {
        // Compute ray origin for new rays spawned from the hit.
        const float3 rayOrigin = sd.computeNewRayOrigin();

        //eval the direct 
        for(int i = 0; i < kLightSamplesPerVertex;i++)
        {
            SceneLightSample ls;
            bool valid = sampleSceneLights(gData.envMapSampler, gData.emissiveSampler, sd, rayOrigin, kLightSamplesPerVertex, rayData.sg, ls);
            if(valid && any(ls.Li > 0.f))
            {
                // Return ray parameters and incident radiance, weighted by path throughput.
                rayData.shadowRay = float4(ls.rayDir, ls.rayDistance);
                float3 test = evalBSDFCosine(sd, ls.dir) * ls.Li;
                if(any(test > 0.f))
                {
                    bool shadowValid = any(test > 0.f);
                    bool V = traceShadowRay(rayOrigin, rayData.shadowRay.xyz, rayData.shadowRay.w, shadowValid);
                    if (V) rayData.Lr += test;
                    //use for debug:the following
                    //rem = vData.normalW;
                }
            }
        }
        // Generate the ray parameters for the next path segment.
        generateScatterRay(sd, rayOrigin, rayData);

        // If path is not terminated, the raygen shader will continue the path based on the returned payload.
        rayData.pathLength++;
    }
}

/** Helper functions for SH evaluation.
*/

void accumShProjection(float3 direction, float value, out float SHcoeffs[81])
{
    for (uint i = 0; i < 81; i++)
        SHcoeffs[i] = 0;

    float x = direction.x;
    float y = direction.y;
    float z = direction.z;
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float x3 = x2 * x;
    float y3 = y2 * y;
    float z3 = z2 * z;
    float x4 = x2 * x2;
    float y4 = y2 * y2;
    float z4 = z2 * z2;
    float x5 = x2 * x3;
    float y5 = y2 * y3;
    float z5 = z2 * z3;
    float x6 = x3 * x3;
    float y6 = y3 * y3;
    float z6 = z3 * z3;
    float x8 = x6 * x2;
    float y8 = y6 * y2;
    float z8 = z6 * z2;

    float cos_term = value;

    SHcoeffs[0] += cos_term * 0.282095;
    SHcoeffs[1] += cos_term * (0.488603 * y);
    SHcoeffs[2] += cos_term * (0.488603 * z);
    SHcoeffs[3] += cos_term * (0.488603 * x);
    SHcoeffs[4] += cos_term * (1.092548 * x * y);
    SHcoeffs[5] += cos_term * (1.092548 * y * z);
    SHcoeffs[6] += cos_term * (0.315392 * (3 * z * z - 1));
    SHcoeffs[7] += cos_term * (1.092548 * x * z);
    SHcoeffs[8] += cos_term * (0.546274 * (x * x - y * y));
    SHcoeffs[9] += cos_term * (0.590043 * (3 * x * x - y * y) * y);
    SHcoeffs[10] += cos_term * (2.89061 * x * y * z);
    SHcoeffs[11] += cos_term * (0.457046 * y * (4 * z * z - x * x - y * y));
    SHcoeffs[12] += cos_term * (0.373176 * (5 * z * z * z - 3 * z));
    SHcoeffs[13] += cos_term * (0.457046 * x * (4 * z * z - x * x - y * y));
    SHcoeffs[14] += cos_term * (1.445307 * (x * x - y * y) * z);
    SHcoeffs[15] += cos_term * (0.590043 * (x * x - 3 * y * y) * x);
    SHcoeffs[16] += cos_term * (2.503342 * (x * y * (x * x - y * y)));
    SHcoeffs[17] += cos_term * (1.770131 * (3 * x * x - y * y) * y * z);
    SHcoeffs[18] += cos_term * (0.946175 * x * y * (7 * z * z - 1));
    SHcoeffs[19] += cos_term * (0.669046 * y * z * (7 * z * z - 3));
    SHcoeffs[20] += cos_term * (0.105786 * (35 * z4 - 30 * z * z + 3));
    SHcoeffs[21] += cos_term * (0.669046 * x * z * (7 * z * z - 3));
    SHcoeffs[22] += cos_term * (0.473087 * (x * x - y * y) * (7 * z * z - 1));
    SHcoeffs[23] += cos_term * (1.770131 * (x * x - 3 * y * y) * x * z);
    SHcoeffs[24] += cos_term * (0.625836 * ((x * x) * (x * x - 3 * y * y) - y * y * (3 * x * x - y * y)));

    // L = 5;
    SHcoeffs[25] += cos_term * (0.6563820568401701 * y * (5. * x4 - 10. * x2 * y2 + y4));
    SHcoeffs[26] += cos_term * (8.302649259524166 * x * (x - 1. * y) * y * (x + y) * z);
    SHcoeffs[27] += cos_term * (0.4892382994352504 * (3. * x2 * y - 1. * y3) * (-1. + 9. * z2));
    SHcoeffs[28] += cos_term * (-4.793536784973324 * x * y * z * (1. - 3. * z2));
    SHcoeffs[29] += cos_term * (0.45294665119569694 * y * (1. - 14. * z2 + 21. * z4));
    SHcoeffs[30] += cos_term * (0.1169503224534236 * z * (15. - 70. * z2 + 63. * z4));
    SHcoeffs[31] += cos_term * (0.45294665119569694 * x * (1. - 14. * z2 + 21. * z4));
    SHcoeffs[32] += cos_term * (2.396768392486662 * (x - 1. * y) * (x + y) * z * (-1. + 3. * z2));
    SHcoeffs[33] += cos_term * (0.4892382994352504 * x * (x2 - 3. * y2) * (-1. + 9. * z2));
    SHcoeffs[34] += cos_term * (2.0756623148810416 * (x4 - 6. * x2 * y2 + y4) * z);
    SHcoeffs[35] += cos_term * (0.6563820568401701 * x * (x4 - 10. * x2 * y2 + 5. * y4));

    // L = 6;
    SHcoeffs[36] += cos_term * (1.3663682103838286 * x * (3. * x4 * y - 10. * x2 * y3 + 3. * y5));
    SHcoeffs[37] += cos_term * (2.366619162231752 * y * (5. * x4 - 10. * x2 * y2 + y4) * z);
    SHcoeffs[38] += cos_term * (-2.0182596029148967 * x * y * (-1. * x + y) * (x + y) * (-1. + 11. * z2));
    SHcoeffs[39] += cos_term * (0.9212052595149236 * (3. * x2 * y - 1. * y3) * z * (-3. + 11. * z2));
    SHcoeffs[40] += cos_term * (0.9212052595149236 * x * y * (1. - 18. * z2 + 33. * z4));
    SHcoeffs[41] += cos_term * (0.5826213625187314 * y * z * (5. - 30. * z2 + 33. * z4));
    SHcoeffs[42] += cos_term * (0.06356920226762842 * (-5. + 21. * z2 * (5. - 15. * z2 + 11. * z4)));
    SHcoeffs[43] += cos_term * (0.5826213625187314 * x * z * (5. - 30. * z2 + 33. * z4));
    SHcoeffs[44] += cos_term * (-0.4606026297574618 * (-1. * x + y) * (x + y) * (1. - 18. * z2 + 33. * z4));
    SHcoeffs[45] += cos_term * (0.9212052595149236 * x * (x2 - 3. * y2) * z * (-3. + 11. * z2));
    SHcoeffs[46] += cos_term * (0.5045649007287242 * (x4 - 6. * x2 * y2 + y4) * (-1. + 11. * z2));
    SHcoeffs[47] += cos_term * (2.366619162231752 * x * (x4 - 10. * x2 * y2 + 5. * y4) * z);
    SHcoeffs[48] += cos_term * (0.6831841051919143 * (x6 - 15. * x4 * y2 + 15. * x2 * y4 - 1. * y6));

    // L=7;
    SHcoeffs[49] += cos_term * (-0.7071627325245962 * y * (-7. * x6 + 35. * x4 * y2 - 21. * x2 * y4 + y6));
    SHcoeffs[50] += cos_term * (5.291921323603801 * x * (3. * x4 * y - 10. * x2 * y3 + 3. * y5) * z);
    SHcoeffs[51] += cos_term * (0.5189155787202604 * y * (5. * x4 - 10. * x2 * y2 + y4) * (-1. + 13. * z2));
    SHcoeffs[52] += cos_term * (4.151324629762083 * x * (x - 1. * y) * y * (x + y) * z * (-3. + 13. * z2));
    SHcoeffs[53] += cos_term * (-0.15645893386229404 * y * (-3. * x2 + y2) * (3. - 66. * z2 + 143. * z4));
    SHcoeffs[54] += cos_term * (0.4425326924449826 * x * y * z * (15. - 110. * z2 + 143. * z4));
    SHcoeffs[55] += cos_term * (0.0903316075825173 * y * (-5. + 135. * z2 - 495. * z4 + 429. * z6));
    SHcoeffs[56] += cos_term * (0.06828427691200495 * z * (-35. + 315. * z2 - 693. * z4 + 429. * z6));
    SHcoeffs[57] += cos_term * (0.0903316075825173 * x * (-5. + 135. * z2 - 495. * z4 + 429. * z6));
    SHcoeffs[58] += cos_term * (0.2212663462224913 * (x - 1. * y) * (x + y) * z * (15. - 110. * z2 + 143. * z4));
    SHcoeffs[59] += cos_term * (0.15645893386229404 * x * (x2 - 3. * y2) * (3. - 66. * z2 + 143. * z4));
    SHcoeffs[60] += cos_term * (1.0378311574405208 * (x4 - 6. * x2 * y2 + y4) * z * (-3. + 13. * z2));
    SHcoeffs[61] += cos_term * (0.5189155787202604 * x * (x4 - 10. * x2 * y2 + 5. * y4) * (-1. + 13. * z2));
    SHcoeffs[62] += cos_term * (2.6459606618019005 * (x6 - 15. * x4 * y2 + 15. * x2 * y4 - 1. * y6) * z);
    SHcoeffs[63] += cos_term * (0.7071627325245962 * x * (x6 - 21. * x4 * y2 + 35. * x2 * y4 - 7. * y6));

    // L = 8;
    SHcoeffs[64] += cos_term * (5.831413281398639 * x * y * (x6 - 7. * x4 * y2 + 7. * x2 * y4 - 1. * y6));
    SHcoeffs[65] += cos_term * (-2.9157066406993195 * y * (-7. * x6 + 35. * x4 * y2 - 21. * x2 * y4 + y6) * z);
    SHcoeffs[66] += cos_term * (1.0646655321190852 * x * (3. * x4 * y - 10. * x2 * y3 + 3. * y5) * (-1. + 15. * z2));
    SHcoeffs[67] += cos_term * (3.449910622098108 * y * (5. * x4 - 10. * x2 * y2 + y4) * z * (-1. + 5. * z2));
    SHcoeffs[68] += cos_term * (1.9136660990373227 * x * (x - 1. * y) * y * (x + y) * (1. - 26. * z2 + 65. * z4));
    SHcoeffs[69] += cos_term * (1.2352661552955442 * (3. * x2 * y - 1. * y3) * z * (3. - 26. * z2 + 39. * z4));
    SHcoeffs[70] += cos_term * (0.912304516869819 * x * y * (-1. + 33. * z2 - 143. * z4 + 143. * z6));
    SHcoeffs[71] += cos_term * (0.10904124589877995 * y * z * (-35. + 11. * z2 * (35. - 91. * z2 + 65. * z4)));
    SHcoeffs[72] += cos_term * (0.009086770491564996 * (35. - 1260. * z2 + 6930. * z4 - 12012. * z6 + 6435. * z8));
    SHcoeffs[73] += cos_term * (0.10904124589877995 * x * z * (-35. + 11. * z2 * (35. - 91. * z2 + 65. * z4)));
    SHcoeffs[74] += cos_term * (0.4561522584349095 * (x - 1. * y) * (x + y) * (-1. + 33. * z2 - 143. * z4 + 143. * z6));
    SHcoeffs[75] += cos_term * (1.2352661552955442 * x * (x2 - 3. * y2) * z * (3. - 26. * z2 + 39. * z4));
    SHcoeffs[76] += cos_term * (0.47841652475933066 * (x4 - 6. * x2 * y2 + y4) * (1. - 26. * z2 + 65. * z4));
    SHcoeffs[77] += cos_term * (3.449910622098108 * x * (x4 - 10. * x2 * y2 + 5. * y4) * z * (-1. + 5. * z2));
    SHcoeffs[78] += cos_term * (0.5323327660595426 * (x6 - 15. * x4 * y2 + 15. * x2 * y4 - 1. * y6) * (-1. + 15. * z2));
    SHcoeffs[79] += cos_term * (2.9157066406993195 * x * (x6 - 21. * x4 * y2 + 35. * x2 * y4 - 7. * y6) * z);
    SHcoeffs[80] += cos_term * (0.7289266601748299 * (x8 - 28. * x6 * y2 + 70. * x4 * y4 - 28. * x2 * y6 + y8));
}

static const float3 zhCenters[17] =
{
    float3(0.866025, -0.500001, -0.000004),
    float3(-0.759553, 0.438522, -0.480394),
    float3(-0.000002, 0.638694, 0.769461),
    float3(-0.000004, -1.000000, -0.000004),
    float3(-0.000007, 0.000003, -1.000000),
    float3(-0.000002, -0.638694, 0.769461),
    float3(-0.974097, 0.000007, -0.226131),
    float3(-0.000003, 0.907079, -0.420960),
    float3(-0.960778, 0.000007, -0.277320),
    float3(-0.000003, -0.907079, -0.420960),
    float3(-0.451627, -0.451622, 0.769461),
    float3(-0.000003, -0.806136, -0.591730),
    float3(0.767864, 0.000000, 0.640613),
    float3(-0.000003, 0.806136, -0.591730),
    float3(-0.553127, 0.319344, 0.769461),
    float3(0.707105, 0.707108, -0.000004),
    float3(0.925820, 0.000000, 0.377964)
};

void zhToSh(const float zhVal[153], out float Lcoeff[81])
{
    float w[17][9];
    for (uint i = 0; i < 17; i++)
        for (uint j = 0; j < 9; j++)
            w[i][j] = zhVal[i * 9 + j];

    for (uint j = 0; j < 81; j++)
        Lcoeff[j] = 0.f;

    Lcoeff[0] = w[0][0];

    Lcoeff[1] = dot(float3(2.1995339, 2.50785367, 1.56572711), float3(w[0][1], w[1][1], w[2][1]));
    Lcoeff[2] = dot(float2(-1.82572523, -2.08165037), float2(w[0][1], w[1][1]));
    Lcoeff[3] = dot(float3(2.42459869, 1.44790525, 0.90397552), float3(w[0][1], w[1][1], w[2][1]));

    Lcoeff[4] = dot(float3(-1.33331385, -0.66666684, -0.99999606), float3(w[0][2], w[3][2], w[4][2]));
    Lcoeff[5] = dot(float3(1.1747938, -0.47923799, -0.69556433), float3(w[2][2], w[3][2], w[4][2]));
    Lcoeff[6] = w[4][2];
    Lcoeff[7] = dot(float3(-1.21710396, 1.58226094, 0.67825711), float3(w[0][2], w[1][2], w[2][2]));
    Lcoeff[7] += dot(float2(-0.27666329, -0.76671491), float2(w[3][2], w[4][2]));
    Lcoeff[8] = dot(float2(-1.15470843, -0.57735948), float2(w[3][2], w[4][2]));
    
    Lcoeff[9] += dot(float3(-0.418128476395, 1.04704832111, 0.418135743058), float3(w[2][3], w[3][3], w[5][3]));
    Lcoeff[10] += dot(float3(-0.217803921828, 1.61365275071, -0.0430709310435), float3(w[0][3], w[1][3], w[2][3]));
    Lcoeff[10] += dot(float3(-1.08141635635, 0.730013109257, -0.906789272616), float3(w[3][3], w[4][3], w[5][3]));
    Lcoeff[11] += dot(float3(0.539792926181, 0.281276817357, -0.53979650602), float3(w[2][3], w[3][3], w[5][3]));
    Lcoeff[12] += -1.00000000026 * w[4][3];
    Lcoeff[13] += dot(float4(-1.88563738164, 0.934959388519, -1.39846078802, -0.934977410564), float4(w[0][3], w[2][3], w[3][3], w[5][3]));
    Lcoeff[14] += dot(float3(-0.822588107798, 0.0250955547337, -0.822583092847), float3(w[2][3], w[4][3], w[5][3]));
    Lcoeff[15] += dot(float3(-1.14577301943, 1.03584677217, -0.849735800355), float3(w[0][3], w[2][3], w[3][3]));
    Lcoeff[15] += dot(float3(-0.438905584229, -0.100364975081, -1.36852983602), float3(w[4][3], w[5][3], w[6][3]));
    Lcoeff[16] += dot(float3(-0.694140591095, -1.46594132085, -3.76291455607), float3(w[0][4], w[1][4], w[2][4]));
    Lcoeff[16] += dot(float3(-4.19771773174, -4.41452625915, -5.21937739623), float3(w[3][4], w[4][4], w[5][4]));
    Lcoeff[16] += dot(float3(30.1096083902, -0.582891410482, -25.58700736), float3(w[6][4], w[7][4], w[8][4]));
    Lcoeff[17] += dot(float4(-0.776237001754, -0.497694700099, 0.155804529921, 0.255292423057), float4(w[2][4], w[3][4], w[4][4], w[5][4]));
    Lcoeff[17] += dot(float3(-0.00123151211175, 0.86352262597, 0.00106337156796), float3(w[6][4], w[7][4], w[8][4]));
    Lcoeff[18] += dot(float3(1.14732747049, -1.93927453351, -4.97819284362), float3(w[0][4], w[1][4], w[2][4]));
    Lcoeff[18] += dot(float3(-4.52057526927, -7.00211058681, -6.90497275343), float3(w[3][4], w[4][4], w[5][4]));
    Lcoeff[18] += dot(float3(39.8336896922, -0.771083185249, -33.8504871326), float3(w[6][4], w[7][4], w[8][4]));
    Lcoeff[19] += dot(float3(0.392392485498, -0.469375435363, 0.146862690526), float3(w[2][4], w[3][4], w[4][4]));
    Lcoeff[19] += dot(float2(-0.883760925422, 0.81431736181), float2(w[5][4], w[7][4]));
    Lcoeff[20] += dot(float3(1.00015572278, -0.00110374505123, 0.000937958411459), float3(w[4][4], w[6][4], w[8][4]));
    Lcoeff[21] += dot(float3(7.51111593422, 6.56318513992, 7.31626822687), float3(w[2][4], w[3][4], w[4][4]));
    Lcoeff[21] += dot(float3(7.51109857163, -51.4260730066, 43.7016908482), float3(w[5][4], w[6][4], w[8][4]));
    Lcoeff[22] += dot(float4(-0.61727564343, 0.205352092062, -0.461764665742, -0.617286413191), float4(w[2][4], w[3][4], w[4][4], w[5][4]));
    Lcoeff[23] += dot(float3(6.71336600734, 5.24419547627, 7.13550000457), float3(w[2][4], w[3][4], w[4][4]));
    Lcoeff[23] += dot(float3(6.71337558899, -51.8339912003, 45.9921960339), float3(w[5][4], w[6][4], w[8][4]));
    Lcoeff[24] += dot(float3(0.466450172383, 1.19684418958, -0.158210638771), float3(w[2][4], w[3][4], w[4][4]));
    Lcoeff[24] += dot(float2(0.466416144347, 0.000906975300098), float2(w[5][4], w[6][4]));

    Lcoeff[25] += dot(float3(0.133023249281, -0.760308430874, -0.132834964007), float3(w[2][5], w[3][5], w[5][5]));
    Lcoeff[25] += dot(float2(0.518450012982, -0.518506289002), float2(w[7][5], w[9][5]));
    Lcoeff[26] += dot(float4(-5.14050071652, 1.83087149155, 1.78238433161, 2.02177436206), float4(w[0][5], w[1][5], w[2][5], w[3][5]));
    Lcoeff[26] += dot(float4(-6.37830163415, 7.15050958772, -11.4689716172, -1.81357297097), float4(w[4][5], w[5][5], w[6][5], w[7][5]));
    Lcoeff[26] += dot(float3(18.797061252, -0.985019094169, 0.739867930848), float3(w[8][5], w[9][5], w[10][5]));
    Lcoeff[27] += dot(float3(-0.805270111174, -0.127526187772, 0.805305675495), float3(w[2][5], w[3][5], w[5][5]));
    Lcoeff[27] += dot(float2(0.0355293099825, -0.0355325587066), float2(w[7][5], w[9][5]));
    Lcoeff[28] += dot(float4(-3.33455180799, 0.473764461749, 1.29183304086, -4.33925590117), float4(w[0][5], w[2][5], w[3][5], w[4][5]));
    Lcoeff[28] += dot(float3(5.75047620831, -11.0487024546, -1.39272172093), float3(w[5][5], w[6][5], w[7][5]));
    Lcoeff[28] += dot(float3(15.5105028631, -1.49843129853, 1.60213069355), float3(w[8][5], w[9][5], w[10][5]));
    Lcoeff[29] += dot(float3(0.677072222771, -0.826014233465, -0.67715296313), float3(w[2][5], w[3][5], w[5][5]));
    Lcoeff[29] += dot(float2(-0.789705431563, 0.789736851279), float2(w[7][5], w[9][5]));
    Lcoeff[30] += -0.999908682153 * w[4][5];
    Lcoeff[31] += dot(float3(3.96619174114, -0.19550135503, -1.58642864103), float3(w[0][5], w[2][5], w[3][5]));
    Lcoeff[31] += dot(float3(3.47400073804, -5.20598671543, 5.5486664469), float3(w[4][5], w[5][5], w[6][5]));
    Lcoeff[31] += dot(float3(0.897822456022, -10.1396069573, 1.11892856733), float3(w[7][5], w[8][5], w[9][5]));
    Lcoeff[32] += dot(float3(-0.616617693524, 0.650557423215, -0.616596023515), float3(w[2][5], w[4][5], w[5][5]));
    Lcoeff[32] += dot(float2(-0.277058316558, -0.277048507641), float2(w[7][5], w[9][5]));
    Lcoeff[33] += dot(float3(5.0039871462, -1.088320635, -2.00153810596), float3(w[0][5], w[2][5], w[3][5]));
    Lcoeff[33] += dot(float3(5.28613602468, -7.40986154696, 16.2642361284), float3(w[4][5], w[5][5], w[6][5]));
    Lcoeff[33] += dot(float3(1.55989830441, -22.7165927272, 1.83889101022), float3(w[7][5], w[8][5], w[9][5]));
    Lcoeff[34] += dot(float3(0.405328070694, -0.0469925687439, 0.405339520518), float3(w[2][5], w[4][5], w[5][5]));
    Lcoeff[34] += dot(float2(-0.608726965055, -0.608728247749), float2(w[7][5], w[9][5]));
    Lcoeff[35] += dot(float3(1.09103970866, -1.17451461211, -0.436442419407), float3(w[0][5], w[2][5], w[3][5]));
    Lcoeff[35] += dot(float3(2.3973180386, -2.55290709278, 3.82908736876), float3(w[4][5], w[5][5], w[6][5]));
    Lcoeff[35] += dot(float3(0.665431405575, -6.99714715269, 0.726274850285), float3(w[7][5], w[8][5], w[9][5]));

    Lcoeff[36] += dot(float4(110.407306498, 121.297251387, 263.985067401, 110.725812649), float4(w[0][6], w[1][6], w[2][6], w[3][6]));
    Lcoeff[36] += dot(float3(322.033930881, 162.479577242, 1755.80451307), float3(w[4][6], w[5][6], w[6][6]));
    Lcoeff[36] += dot(float3(-14.7032957811, -1553.88668558, 64.4632133415), float3(w[7][6], w[8][6], w[9][6]));
    Lcoeff[36] += dot(float3(-22.0873701656, -61.3998433111, 187.359713154), float3(w[10][6], w[11][6], w[12][6]));
    Lcoeff[37] += dot(float4(-0.00111818967084, -0.00123422458098, 0.336682549226, -0.00132691821183), float4(w[0][6], w[1][6], w[2][6], w[3][6]));
    Lcoeff[37] += dot(float4(-0.0034983376219, -0.341167543299, -0.01890204627, -0.723728647079), float4(w[4][6], w[5][6], w[6][6], w[7][6]));
    Lcoeff[37] += dot(float3(0.0167673427229, 0.723491153889, -0.00198206765087), float3(w[8][6], w[9][6], w[12][6]));
    Lcoeff[38] += dot(float4(15.3227047788, 14.2753461798, 12.4149247001, 1.340434285), float4(w[0][6], w[1][6], w[2][6], w[3][6]));
    Lcoeff[38] += dot(float3(16.4050796616, 5.20690955596, 93.7391441162), float3(w[4][6], w[5][6], w[6][6]));
    Lcoeff[38] += dot(float3(-5.70151762203, -78.7068450207, 1.01952589216), float3(w[7][6], w[8][6], w[9][6]));
    Lcoeff[38] += dot(float3(-0.0124081990428, -2.71358001541, 13.8096132986), float3(w[10][6], w[11][6], w[12][6]));
    Lcoeff[39] += dot(float4(-0.00117951166106, -1.68851028063, 0.499987655626, 0.0811287014828), float4(w[1][6], w[2][6], w[3][6], w[4][6]));
    Lcoeff[39] += dot(float4(0.924525864736, -0.0180671469572, -0.53953363757, 0.0160213923983), float4(w[5][6], w[6][6], w[7][6], w[8][6]));
    Lcoeff[39] += dot(float3(-1.22120890317, 1.93471815616, -0.00189817210815), float3(w[9][6], w[11][6], w[12][6]));
    Lcoeff[40] += dot(float4(14.2349433166, 15.6374252438, 13.8493129894, 2.43692566103), float4(w[0][6], w[1][6], w[2][6], w[3][6]));
    Lcoeff[40] += dot(float3(16.7738688981, 5.95348061509, 102.676997666), float3(w[4][6], w[5][6], w[6][6]));
    Lcoeff[40] += dot(float3(-4.85746148998, -86.2109081623, 2.50489677852), float3(w[7][6], w[8][6], w[9][6]));
    Lcoeff[40] += dot(float3(-0.0135530622167, -2.97256381466, 15.1268484906), float3(w[10][6], w[11][6], w[12][6]));
    Lcoeff[41] += dot(float4(0.0011471762337, 0.00134013654614, 1.90642287302, -0.949799062166), float4(w[0][6], w[1][6], w[2][6], w[3][6]));
    Lcoeff[41] += dot(float4(-0.156687609791, -0.460381337914, 0.0194055069076, 0.614031832908), float4(w[4][6], w[5][6], w[6][6], w[7][6]));
    Lcoeff[41] += dot(float4(-0.017163787015, 2.72661250835, -3.67080744599, 0.00207762117982), float4(w[8][6], w[9][6], w[11][6], w[12][6]));
    Lcoeff[42] += dot(float3(1.00039252313, 0.00181905245706, -0.00166508494552), float3(w[4][6], w[6][6], w[8][6]));
    Lcoeff[43] += dot(float4(0.0133463019389, 0.0153654992493, 8.78743258451, 5.81522851174), float4(w[0][6], w[1][6], w[2][6], w[3][6]));
    Lcoeff[43] += dot(float3(12.5550974713, 8.77413125608, 58.4922368204), float3(w[4][6], w[5][6], w[6][6]));
    Lcoeff[43] += dot(float3(1.34231351205, -53.541181195, 1.35225002587), float3(w[7][6], w[8][6], w[9][6]));
    Lcoeff[43] += dot(float3(-0.0032149390263, -0.00792693014645, 3.26902843874), float3(w[10][6], w[11][6], w[12][6]));
    Lcoeff[44] += dot(float3(-0.565026447733, -0.440924426872, -0.803140025769), float3(w[2][6], w[3][6], w[4][6]));
    Lcoeff[44] += dot(float3(-0.565750951282, 0.0106971854579, 0.465219496817), float3(w[5][6], w[6][6], w[7][6]));
    Lcoeff[44] += dot(float3(-0.00942810590012, 0.465803316978, 0.00121182174646), float3(w[8][6], w[9][6], w[12][6]));
    Lcoeff[45] += dot(float4(0.0120591281871, 0.0140081081238, 8.84231008623, 6.30335116571), float4(w[0][6], w[1][6], w[2][6], w[3][6]));
    Lcoeff[45] += dot(float3(13.3925214762, 8.83016166486, 63.6145973482), float3(w[4][6], w[5][6], w[6][6]));
    Lcoeff[45] += dot(float3(1.11030347967, -59.2232957105, 1.11930309013), float3(w[7][6], w[8][6], w[9][6]));
    Lcoeff[45] += dot(float3(-0.00299575240939, -0.00721839822023, 4.87051865893), float3(w[10][6], w[11][6], w[12][6]));
    Lcoeff[46] += dot(float3(0.651047411165, -0.429335217103, 0.106976359799), float3(w[2][6], w[3][6], w[4][6]));
    Lcoeff[46] += dot(float3(0.650368759846, 0.0109636003151, 0.306691437259), float3(w[5][6], w[6][6], w[7][6]));
    Lcoeff[46] += dot(float3(-0.00965411398733, 0.307235947971, 0.00120365403511), float3(w[8][6], w[9][6], w[12][6]));
    Lcoeff[47] += dot(float4(0.00324439166172, 0.00369123568018, 0.800772124437, 1.63137398167), float4(w[0][6], w[1][6], w[2][6], w[3][6]));
    Lcoeff[47] += dot(float4(1.43566157126, 0.797501947007, 9.03263537993, 0.535950980597), float4(w[4][6], w[5][6], w[6][6], w[7][6]));
    Lcoeff[47] += dot(float4(-7.2757158051, 0.538410039638, -0.00198357688491, 1.7022272091), float4(w[8][6], w[9][6], w[11][6], w[12][6]));
    Lcoeff[48] += dot(float4(0.00156677824614, 0.00170502607019, -0.0939749600966, -0.872048853744), float4(w[0][6], w[1][6], w[2][6], w[3][6]));
    Lcoeff[48] += dot(float4(0.00440208146779, -0.0953605716075, 0.0240666470029, -0.540590886575), float4(w[4][6], w[5][6], w[6][6], w[7][6]));
    Lcoeff[48] += dot(float3(-0.0212662429579, -0.539478586759, 0.00260075917574), float3(w[8][6], w[9][6], w[12][6]));

    Lcoeff[49] += dot(float3(-0.0701805272893, 0.98255354265, 0.0686440478357), float3(w[2][7], w[3][7], w[5][7]));
    Lcoeff[49] += dot(float3(-0.550199211871, 0.550738563391, -0.00124904263303), float3(w[7][7], w[9][7], w[13][7]));
    Lcoeff[50] += dot(float4(-37.3285759367, -5.7563959375, 44.5510845679, 53.5563319108), float4(w[0][7], w[2][7], w[3][7], w[4][7]));
    Lcoeff[50] += dot(float4(-114.931760993, 26.7620910072, 76.1871627236, 18.7459525504), float4(w[5][7], w[6][7], w[7][7], w[8][7]));
    Lcoeff[50] += dot(float3(-32.9000483785, -7.89959890218, 45.2203492106), float3(w[9][7], w[10][7], w[11][7]));
    Lcoeff[50] += dot(float3(24.5638783781, -109.38781242, 0.00341946781379), float3(w[12][7], w[13][7], w[14][7]));
    Lcoeff[51] += dot(float3(0.60312464974, 0.514140412839, -0.601853534614), float3(w[2][7], w[3][7], w[5][7]));
    Lcoeff[51] += dot(float2(0.456751969399, -0.457204648754), float2(w[7][7], w[9][7]));
    Lcoeff[52] += dot(float4(-5.52955347687, -4.34348722462, 4.3438125575, 7.88948108037), float4(w[0][7], w[2][7], w[3][7], w[4][7]));
    Lcoeff[52] += dot(float3(-12.6234155916, 6.34843469899, 6.2109893889), float3(w[5][7], w[6][7], w[7][7]));
    Lcoeff[52] += dot(float3(-0.00429379248923, -0.756126464499, 0.424399863695), float3(w[8][7], w[9][7], w[11][7]));
    Lcoeff[52] += dot(float3(3.60062053248, -9.77362846891, -2.02096723974), float3(w[12][7], w[13][7], w[14][7]));
    Lcoeff[53] += dot(float4(-1.84219028026, -0.800561174358, 1.84228914268, -1.76578138065), float4(w[2][7], w[3][7], w[5][7], w[7][7]));
    Lcoeff[53] += dot(float3(1.76570658332, -2.58447322137, 2.58446953723), float3(w[9][7], w[11][7], w[13][7]));
    Lcoeff[54] += dot(float4(4.28585473464, -5.57083870671, 3.88299131276, -1.21598103809), float4(w[0][7], w[1][7], w[2][7], w[3][7]));
    Lcoeff[54] += dot(float4(-1.02579013048, 0.421658914765, -6.27175023832, 1.89012154573), float4(w[4][7], w[5][7], w[6][7], w[7][7]));
    Lcoeff[54] += dot(float3(0.0116877614083, -3.62740434708, -0.00168583412282), float3(w[8][7], w[9][7], w[10][7]));
    Lcoeff[54] += dot(float3(2.90947495533, 2.71420375237, -1.15571591609), float3(w[11][7], w[12][7], w[13][7]));
    Lcoeff[55] += dot(float4(1.33143227971, 1.12171235628, -1.33021306817, 2.17150195324), float4(w[2][7], w[3][7], w[5][7], w[7][7]));
    Lcoeff[55] += dot(float3(-2.17190603892, 2.68546413814, -2.68475126266), float3(w[9][7], w[11][7], w[13][7]));
    Lcoeff[56] += -0.999785087625 * w[4][7];
    Lcoeff[57] += dot(float4(-4.98353234343, -0.60335722233, 5.06180432247, 5.983807289), float4(w[0][7], w[2][7], w[3][7], w[4][7]));
    Lcoeff[57] += dot(float4(-12.4335970683, 5.62509069724, 8.45051745261, -0.00408078300172), float4(w[5][7], w[6][7], w[7][7], w[8][7]));
    Lcoeff[57] += dot(float4(-2.8882608379, 4.59925710784, 2.29940628779, -11.9972155397), float4(w[9][7], w[11][7], w[12][7], w[13][7]));
    Lcoeff[58] += dot(float4(-0.87292844039, 0.853086268401, -0.872716127179, 0.548554049149), float4(w[2][7], w[4][7], w[5][7], w[7][7]));
    Lcoeff[58] += dot(float3(0.548734794611, -1.07309588446, -1.07281880178), float3(w[9][7], w[11][7], w[13][7]));
    Lcoeff[59] += dot(float4(-6.71247285675, -2.43207100005, 6.81762410543, 9.33449029193), float4(w[0][7], w[2][7], w[3][7], w[4][7]));
    Lcoeff[59] += dot(float3(-18.3657135974, 2.02534151283, 11.8803536762), float3(w[5][7], w[6][7], w[7][7]));
    Lcoeff[59] += dot(float3(6.40090944988, -3.39122648096, 0.00113658729748), float3(w[8][7], w[9][7], w[10][7]));
    Lcoeff[59] += dot(float3(5.38458251565, 4.50642495235, -16.9684952577), float3(w[11][7], w[12][7], w[13][7]));
    Lcoeff[60] += dot(float3(0.8151794721, -0.201027115556, 0.81519502336), float3(w[2][7], w[4][7], w[5][7]));
    Lcoeff[60] += dot(float2(0.181660069418, 0.181680334223), float2(w[7][7], w[9][7]));
    Lcoeff[61] += dot(float4(-1.90839389632, -1.76712931026, 1.93832899575, 3.89588368627), float4(w[0][7], w[2][7], w[3][7], w[4][7]));
    Lcoeff[61] += dot(float4(-6.29727952971, 2.48144221527, 3.24427850885, -1.09768856807), float4(w[5][7], w[6][7], w[7][7], w[9][7]));
    Lcoeff[61] += dot(float3(0.748671845519, 2.61667793247, -5.60663032021), float3(w[11][7], w[12][7], w[13][7]));
    Lcoeff[62] += dot(float4(-0.270895144417, -0.271153759143, 0.820526195092, 0.820278327508), float4(w[2][7], w[5][7], w[7][7], w[9][7]));
    Lcoeff[63] += dot(float4(-0.00126785006862, 1.20665045598, 0.00162281551801, -0.963247933527), float4(w[0][7], w[2][7], w[3][7], w[4][7]));
    Lcoeff[63] += dot(float4(1.20267970448, -1.77208455178, -0.986776039728, 0.0024149973414), float4(w[5][7], w[6][7], w[7][7], w[8][7]));
    Lcoeff[63] += dot(float4(-0.990721508828, 0.583109464442, 0.451361607718, 0.577458802184), float4(w[9][7], w[11][7], w[12][7], w[13][7]));

    Lcoeff[64] += dot(float4(1.53749427566, 0.909584774471, -12.8120323661, 0.431904370785), float4(w[0][8], w[1][8], w[2][8], w[3][8]));
    Lcoeff[64] += dot(float4(-12.3258435464, -13.1146498411, 88.2060200532, -12.2132842255), float4(w[4][8], w[5][8], w[6][8], w[7][8]));
    Lcoeff[64] += dot(float3(-111.819986738, -11.6817657491, 0.404047756365), float3(w[8][8], w[9][8], w[10][8]));
    Lcoeff[64] += dot(float3(12.0467189584, -0.994477402231, 12.180903566), float3(w[11][8], w[12][8], w[13][8]));
    Lcoeff[64] += dot(float3(-0.509165463671, 0.214834626621, 32.2725793151), float3(w[14][8], w[15][8], w[16][8]));
    Lcoeff[65] += dot(float4(0.02820380227, -0.0747431249236, 2.92512698977, -0.112167836437), float4(w[0][8], w[1][8], w[2][8], w[3][8]));
    Lcoeff[65] += dot(float4(2.79794117593, 3.55063984248, -21.5072045956, 3.93193106867), float4(w[4][8], w[5][8], w[6][8], w[7][8]));
    Lcoeff[65] += dot(float3(27.3991444823, 1.94595833674, 0.0115842799386), float3(w[8][8], w[9][8], w[10][8]));
    Lcoeff[65] += dot(float3(-2.64814758249, 0.42709474181, -2.94245915297), float3(w[11][8], w[12][8], w[13][8]));
    Lcoeff[65] += dot(float3(-0.0686794362676, 0.00655079684495, -7.93317808252), float3(w[14][8], w[15][8], w[16][8]));
    Lcoeff[66] += dot(float4(0.391342450648, -1.02492254371, 125.326506712, -4.6445432842), float4(w[0][8], w[1][8], w[2][8], w[3][8]));
    Lcoeff[66] += dot(float4(109.33702209, 125.208534258, -851.51589026, 114.502675185), float4(w[4][8], w[5][8], w[6][8], w[7][8]));
    Lcoeff[66] += dot(float3(1087.75897489, 113.224769946, -1.50797447135), float3(w[8][8], w[9][8], w[10][8]));
    Lcoeff[66] += dot(float3(-108.466362134, 18.4164681113, -109.583142595), float3(w[11][8], w[12][8], w[13][8]));
    Lcoeff[66] += dot(float3(-0.937471108497, 1.39128181347, -316.5905216), float3(w[14][8], w[15][8], w[16][8]));
    Lcoeff[67] += dot(float4(0.119153299569, -0.315481464262, 14.0614420783, -0.473010097461), float4(w[0][8], w[1][8], w[2][8], w[3][8]));
    Lcoeff[67] += dot(float4(11.7965154114, 13.2422133423, -90.6761509524, 12.8951524656), float4(w[4][8], w[5][8], w[6][8], w[7][8]));
    Lcoeff[67] += dot(float3(115.516527355, 11.8873201428, 0.049144869583), float3(w[8][8], w[9][8], w[10][8]));
    Lcoeff[67] += dot(float3(-11.1652789635, 1.80037152648, -12.4058801396), float3(w[11][8], w[12][8], w[13][8]));
    Lcoeff[67] += dot(float3(-0.289812743287, 0.0276724657153, -33.4465000918), float3(w[14][8], w[15][8], w[16][8]));
    Lcoeff[68] += dot(float4(-0.0291848617478, 0.245516477549, 118.116347051, -3.38172094147), float4(w[0][8], w[1][8], w[2][8], w[3][8]));
    Lcoeff[68] += dot(float4(104.366969862, 117.524021629, -813.126174282, 108.739088105), float4(w[4][8], w[5][8], w[6][8], w[7][8]));
    Lcoeff[68] += dot(float3(1040.02520445, 108.127023707, -1.06230167797), float3(w[8][8], w[9][8], w[10][8]));
    Lcoeff[68] += dot(float3(-102.174880088, 18.9774125832, -103.213055714), float3(w[11][8], w[12][8], w[13][8]));
    Lcoeff[68] += dot(float3(-1.81889823372, -0.555751123944, -303.112465503), float3(w[14][8], w[15][8], w[16][8]));
    Lcoeff[69] += dot(float4(-0.395035183387, 1.04586983477, -45.5572484204, 1.56813129402), float4(w[0][8], w[1][8], w[2][8], w[3][8]));
    Lcoeff[69] += dot(float4(-39.1074829737, -44.9591485736, 300.607189459, -41.1873913838), float4(w[4][8], w[5][8], w[6][8], w[7][8]));
    Lcoeff[69] += dot(float3(-382.957356484, -40.9708031646, -0.162930068218), float3(w[8][8], w[9][8], w[10][8]));
    Lcoeff[69] += dot(float3(39.2832393404, -5.96855234135, 38.8592527753), float3(w[11][8], w[12][8], w[13][8]));
    Lcoeff[69] += dot(float3(0.960789726517, -0.0917705056566, 110.880958708), float3(w[14][8], w[15][8], w[16][8]));
    Lcoeff[70] += dot(float4(0.456810706057, -1.19610219984, 145.14169054, -3.60731438449), float4(w[0][8], w[1][8], w[2][8], w[3][8]));
    Lcoeff[70] += dot(float4(128.010827764, 145.004047784, -993.769284631, 134.16474103), float4(w[4][8], w[5][8], w[6][8], w[7][8]));
    Lcoeff[70] += dot(float3(1269.47881154, 132.673316481, -1.75982783541), float3(w[8][8], w[9][8], w[10][8]));
    Lcoeff[70] += dot(float3(-125.717600801, 21.4930188871, -127.020961656), float3(w[11][8], w[12][8], w[13][8]));
    Lcoeff[70] += dot(float3(-1.09412783044, -0.92615123861, -369.479614638), float3(w[14][8], w[15][8], w[16][8]));
    Lcoeff[71] += dot(float4(0.367481338147, -0.972917666468, 41.653343463, -1.45876523059), float4(w[0][8], w[1][8], w[2][8], w[3][8]));
    Lcoeff[71] += dot(float4(36.3807277786, 42.5517382838, -279.647500028, 38.6375231848), float4(w[4][8], w[5][8], w[6][8], w[7][8]));
    Lcoeff[71] += dot(float3(356.255877922, 37.7922040017, 0.151565737021), float3(w[8][8], w[9][8], w[10][8]));
    Lcoeff[71] += dot(float3(-35.8358023732, 5.55242780488, -36.8581681038), float3(w[11][8], w[12][8], w[13][8]));
    Lcoeff[71] += dot(float3(-0.893804648761, 0.0853291679856, -103.149863518), float3(w[14][8], w[15][8], w[16][8]));
    Lcoeff[72] += dot(float4(0.00188542464872, 1.00165569552, 0.00187041518417, -0.0123197787928), float4(w[2][8], w[4][8], w[5][8], w[6][8]));
    Lcoeff[72] += dot(float3(0.00169115843328, 0.0156826399574, 0.00170503250921), float3(w[7][8], w[8][8], w[9][8]));
    Lcoeff[72] += dot(float3(-0.00167433376791, -0.00165711826159, -0.00449049345544), float3(w[11][8], w[13][8], w[16][8]));
    Lcoeff[73] += dot(float4(0.00405256436808, -0.00262582264354, 63.550287794, -2.30613108952), float4(w[0][8], w[1][8], w[2][8], w[3][8]));
    Lcoeff[73] += dot(float4(55.7252304106, 63.5538445478, -418.347495226, 57.5308076948), float4(w[4][8], w[5][8], w[6][8], w[7][8]));
    Lcoeff[73] += dot(float4(532.655148546, 57.521374147, -0.00352791192438, -55.9572413108), float4(w[8][8], w[9][8], w[10][8], w[11][8]));
    Lcoeff[73] += dot(float4(7.99539527162, -55.9579600129, 0.00456856106269, -152.990616437), float4(w[12][8], w[13][8], w[15][8], w[16][8]));
    Lcoeff[74] += dot(float4(-0.881394274788, 0.473165408429, -0.942062797089, -0.881378435578), float4(w[2][8], w[3][8], w[4][8], w[5][8]));
    Lcoeff[74] += dot(float4(0.0177306070586, -1.01197251544, -0.0226882112384, -1.0120079567), float4(w[6][8], w[7][8], w[8][8], w[9][8]));
    Lcoeff[74] += dot(float3(1.4069429521, 1.40690203994, 0.00662191089822), float3(w[11][8], w[13][8], w[16][8]));
    Lcoeff[75] += dot(float4(0.00641782135735, -0.00388890880837, 92.7897539783, -2.83263971836), float4(w[0][8], w[1][8], w[2][8], w[3][8]));
    Lcoeff[75] += dot(float4(81.5477534167, 92.7953548562, -635.681288259, 84.4996736106), float4(w[4][8], w[5][8], w[6][8], w[7][8]));
    Lcoeff[75] += dot(float4(814.870318064, 84.4851189109, -0.00533844276317, -81.4656461897), float4(w[8][8], w[9][8], w[10][8], w[11][8]));
    Lcoeff[75] += dot(float4(13.9482979732, -81.4667603321, 0.00699793551893, -238.52207368), float4(w[12][8], w[13][8], w[15][8], w[16][8]));
    Lcoeff[76] += dot(float4(1.17167184082, -0.106264795054, 0.271482557427, 1.17166744313), float4(w[2][8], w[3][8], w[4][8], w[5][8]));
    Lcoeff[76] += dot(float4(-0.00997814053713, 0.393888061274, 0.0127991226512, 0.393896718706), float4(w[6][8], w[7][8], w[8][8], w[9][8]));
    Lcoeff[76] += dot(float3(-0.89395978918, -0.893949593311, -0.00378935185386), float3(w[11][8], w[13][8], w[16][8]));
    Lcoeff[77] += dot(float4(0.0026121874954, -0.00155417390692, 34.9602498365, -0.615998044572), float4(w[0][8], w[1][8], w[2][8], w[3][8]));
    Lcoeff[77] += dot(float4(30.2173293689, 34.9625301883, -252.286117615, 32.1648727481), float4(w[4][8], w[5][8], w[6][8], w[7][8]));
    Lcoeff[77] += dot(float4(324.269439477, 32.1590190028, -0.00211835901653, -29.8023247437), float4(w[8][8], w[9][8], w[10][8], w[11][8]));
    Lcoeff[77] += dot(float4(7.39874675695, -29.8028199408, 0.00279330634013, -96.341082571), float4(w[12][8], w[13][8], w[15][8], w[16][8]));
    Lcoeff[78] += dot(float4(-0.387527707899, 0.416763243365, -0.0286688282881, -0.387525584935), float4(w[2][8], w[3][8], w[4][8], w[5][8]));
    Lcoeff[78] += dot(float4(-0.0105262321023, -0.295992700404, 0.0134560889041, -0.296013229931), float4(w[6][8], w[7][8], w[8][8], w[9][8]));
    Lcoeff[78] += dot(float3(-0.345151726145, -0.345174064069, -0.00393134444984), float3(w[11][8], w[13][8], w[16][8]));
    Lcoeff[79] += dot(float4(4.90103610639, -1.21333206278, 4.02714177428, 4.90129554805), float4(w[2][8], w[3][8], w[4][8], w[5][8]));
    Lcoeff[79] += dot(float4(-28.4484536271, 3.91036185913, 37.1656352206, 3.90974932967), float4(w[6][8], w[7][8], w[8][8], w[9][8]));
    Lcoeff[79] += dot(float4(-3.90148035383, 1.47547837629, -3.90145837153, -9.91757330562), float4(w[11][8], w[12][8], w[13][8], w[16][8]));
    Lcoeff[80] += dot(float4(0.0684332108322, 1.06483675683, -0.00153837235285, 0.0684374426626), float4(w[2][8], w[3][8], w[4][8], w[5][8]));
    Lcoeff[80] += dot(float4(-0.0166644698234, 0.593587849063, 0.0213264820211, 0.593536664019), float4(w[6][8], w[7][8], w[8][8], w[9][8]));
    Lcoeff[80] += dot(float3(-0.0440627637216, -0.0441185577727, -0.00620665050047), float3(w[11][8], w[13][8], w[16][8]));
}

/** ********************* Ray index 0: Scatter ray ************************ */

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData : SV_RayPayload)
{
    handleMiss(gData, rayData);
}

[shader("anyhit")]
void scatterAnyHit(
    uniform HitShaderParams hitParams,
    inout ScatterRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    VertexData v = getVertexData(hitParams, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
}

[shader("closesthit")]
void scatterClosestHit(
    uniform HitShaderParams hitParams,
    inout ScatterRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    //get the data about the hit point
    const float3 rayDir = WorldRayDirection();
    VertexData v = getVertexData(hitParams,PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());
    ShadingData sd = prepareShadingData(v, materialID,gScene.materials[materialID], gScene.materialResources[materialID], -WorldRayDirection(), 0.f);

    // Compute tangent space if it is invalid.
    if (!(dot(sd.T, sd.T) > 0.f))   // Note: Comparison written so that NaNs trigger
    {
        sd.T = perp_stark(sd.N);
        sd.B = cross(sd.N, sd.T);
    }

    
    // These fields are needed for ray offset computatation and light PDF evaluation.
    Trianglehit hit;
    hit.meshInstanceID = hitParams.getGlobalHitID();
    hit.primitiveIndex = PrimitiveIndex();
    hit.posW = sd.posW;
    hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;
    hit.triangleArea = gScene.getFaceAreaW(hit.meshInstanceID, hit.primitiveIndex);

    handleHit(gData, sd, hit, kComputeIndirect, rayData);
    //rayData.packedHitInfo = hit.encode();
}

/************************** Ray index 1: Shadow ray ************************ */

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData : SV_RayPayload)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

[shader("anyhit")]
void shadowAnyHit(
    uniform HitShaderParams hitParams,
    inout ShadowRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    VertexData v = getVertexData(hitParams, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
}


/** ******************************** RayGen ******************************** */

/** This is the entry point for the path tracer.

    We generate N paths (= #spp) per pixel, which are traced into the scene.
    The path tracer is written as a for-loop over path segments, where each
    iteration traces a shadow ray for direct illumination and a scatter ray.

    The hit shader for the scatter ray currently generates ray parameters for
    the shadow ray to evaluate direct illumination and generates ray parameters
    for the next scatter ray, which are both returned the raygen shader to be
    traced. This is more efficient than tracing from the hit shader. The max
    recusion depth = 1.
*/
[shader("raygeneration")]
void rayGen()
{
    uint vertexID = gVertexCount;
    //uint vertexID = 55700;
    float3 rem = float3(0);
    uint gPRNGDimension = 0;
    SampleGenerator sg = SampleGenerator.create(DispatchRaysIndex().xy,vertexID);
    
    uint2 vInfo = gVertexToMesh[vertexID];
    uint materialID = gScene.getMaterialID(vInfo.x);

    uint meshInstanceID = vInfo.x;
    uint triangleID = vInfo.y;

    float delta = 1e-2;

    uint y = DispatchRaysIndex().y;
    uint x = DispatchRaysIndex().x;
    
    float3 outDirect = float3(0, 0, 0);
    float3 outIndirect = float3(0, 0, 0);

    float3 outColor = float3(0, 0, 0);
    uint SPP = kSamplesPerPixel;
    float3 mark;
    float3 barycentrics = float3(0.33,0.33,0.34);
    VertexData vData = gScene.getVertexData(meshInstanceID, triangleID, barycentrics);
    float3 bitangentW = cross(vData.normalW, vData.tangentW.xyz);
    const uint kLastBounce = kComputeIndirect ? kMaxBounces : 0;
    const uint kIterationCount = kLastBounce;
    for(uint j = 0;j < SPP;j++){
        
        float2 rnd = sampleNext2D(sg);
        float u = (float(x) + rnd.x) / float(guSpp);
        float v = (float(y) + rnd.y) / float(gvSpp);
        float pdf;
        float3 dirLocal = sample_cosine_hemisphere_polar(float2(u, v), pdf);
        
        float3 rayDir_inv = dirLocal.x * vData.tangentW.xyz + dirLocal.y * bitangentW + dirLocal.z * vData.normalW;
        rayDir_inv = normalize(rayDir_inv);
        float3 worldPos = vData.posW + rayDir_inv * delta;

        float3 rayDir = -rayDir_inv;

        float3 Ldirect = float3(0, 0, 0);
        float3 Lindirect = float3(0, 0, 0);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++) sampleNext1D(sg);

        // Prepare ray payload.
        ScatterRayData rayData = ScatterRayData.create(sg);

        //rayDir反转有问题吗！？
        ShadingData sd = prepareShadingData(vData, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -rayDir, 0.f);
        // Compute tangent space if it is invalid.
        if (!(dot(sd.T, sd.T) > 0.f))   // Note: Comparison written so that NaNs trigger
        {
            sd.T = perp_stark(sd.N);
            sd.B = cross(sd.N, sd.T);
        }
        
        // Generate scatter ray.
        rayData.terminated = false;
        rayData.pathLength = 0;
        rayData.direction = rayDir;
        rayData.thp = float3(1, 1, 1);
        rayData.normal = float3(0, 0, 0);
        rayData.pdf = 1.f;
        float3 rayOrigin = sd.computeNewRayOrigin();
        //float3 rayOrigin = vData.posW;
        
        rayData.origin = rayOrigin;
        if (any(sd.emissive > 0.f))
        {
            float misWeight = 1.f;
            if (kUseEmissiveSampler && kUseMIS)
            {
                // If emissive light sampling and MIS is enabled, we've already sampled emissive lights using NEE.
                // We need to evaluate the MIS weight here to account for the remaining contribution.

                // Evaluate PDF at the hit, had it been generated with light sampling.
                // Note that there is a minor discrepancy compared to BRDF sampling at the previous path vertex:
                // the normal passed through the ray payload is lossily compressed to reduce the ray payload size.
                TriangleHit Thit;
                Thit.triangleIndex = gScene.lightCollection.getTriangleIndex(meshInstanceID, triangleID);
                Thit.posW = vData.posW;
                Thit.normalW = vData.normalW;

                float lightPdf = gData.emissiveSampler.evalPdf(rayData.origin, rayData.normal, Thit) * getEmissiveLightSelectionPdf();

                // Compute MIS weight by combining this with BRDF sampling.
                // Note we can assume rayData.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(rayData.pdf, lightPdf);
            }

            // Return emitted radiance weighted by path throughput and MIS weight.
            Ldirect += rayData.thp * sd.emissive * misWeight;
        }

        // Generate the ray parameters for the next path segment.
        for(int i = 0; i < kLightSamplesPerVertex;i++)
        {
            SceneLightSample ls;
            bool valid = sampleSceneLights(gData.envMapSampler, gData.emissiveSampler, sd, rayOrigin, kLightSamplesPerVertex, rayData.sg, ls);
            if(valid && any(ls.Li > 0.f))
            {
                // Return ray parameters and incident radiance, weighted by path throughput.
                rayData.shadowRay = float4(ls.rayDir, ls.rayDistance);
                float3 test = evalBSDFCosine(sd, ls.dir) * ls.Li;
                if(any(test > 0.f))
                {
                    bool shadowValid = any(test > 0.f);
                    bool V = traceShadowRay(rayOrigin, rayData.shadowRay.xyz, rayData.shadowRay.w, shadowValid);
                    if (V) Lindirect += test;
                    //use for debug:the following
                    //rem = vData.normalW;
                }
            }
        }
        //Lindirect = Lindirect / kLightSamplesPerVertex;
        generateScatterRay(sd, rayOrigin, rayData);
        
        [unroll]  
        for (uint depth = 0; depth < kIterationCount && !rayData.terminated; depth++)
        //for (uint depth = 0; depth < 0; depth++)
        {
                // Reset contributions.
                rayData.Le = rayData.Lr = float3(0, 0, 0);
                
                
                // Trace scatter ray. The closest hit shader generates a shadow ray and a new scatter ray.
                traceScatterRay(rayData);
                //if (!rayData.terminated) Ldirect = float3(rayData.pdf);
                //else Ldirect = float3(0);
                // Accumulate emitted radiance as direct/indirect depending on path length.
                Lindirect += rayData.Le;
                if ((depth == 0) && (!rayData.terminated)) {
                    gOutGoingNormal[uint2(x,y)] = float4(rayData.normal, 1);
                    float3 dist = rayData.origin - vData.posW;
                    float dist_val = sqrt(dot(dist, dist));
                    //gOutGoingDistance[uint2(x,y)] = float4(dist, 1);
                    gOutGoingDistance[uint2(x,y)] = float4(dist_val,dist_val,dist_val,1);
                }
                else{
                    gOutGoingNormal[uint2(x,y)] = float4(float3(0), 1);
                    gOutGoingDistance[uint2(x,y)] = float4(float3(0),1);
                }
                if (depth < kLastBounce)
                {
                    // Trace shadow ray and accumulate reflected radiance as indirect if light is visible.
                    Lindirect += rayData.Lr;

                    // Russian roulette to stochastically terminate the path.
                    // We use a fixed absorption probability for now.
                    // TODO: Better strategy, e.g., 1-P(absorption) = hemispherical reflectance of BRDF.
                    if (kUseRussianRoulette)
                    {
                        float u = sampleNext1D(rayData.sg);
                        if (u < probabilityAbsorption) rayData.terminated = true;
                        rayData.thp /= (1.f - probabilityAbsorption);
                    }
                }
        }

        // Accumulate direct/indirect illumination after clamping.
        // Note the comparisons are purposely written so that NaNs propagate (unless the compiler rewrites it).
        //clamp有点问题
        outDirect += clampDirect && Ldirect > thresholdDirect ? thresholdDirect : Ldirect;
        outIndirect += clampIndirect && Lindirect > thresholdIndirect ? thresholdIndirect : Lindirect;
        
        //outDirect += Ldirect;
        //outIndirect += Lindirect;
    }
    outColor = (outDirect + outIndirect) / SPP;
    gOutGoingRadiance[uint2(x,y)] = float4(outColor,1);

    //for debug
    //gOutGoingRadiance[uint2(x,y)] = float4(vData.normalW,1);

    float SHcoeffs[81];
    //rayDir is in the world coordinate
    //for R part
    float u = float(x) / float(guSpp);
    float v = float(y) / float(gvSpp);
    float pdf;
    float3 dirLocal = sample_cosine_hemisphere_polar(float2(u, v), pdf);
        
    float3 rayDir_inv = dirLocal.x * vData.tangentW.xyz + dirLocal.y * bitangentW + dirLocal.z * vData.normalW;
    float3 rayDir = -rayDir_inv;
    accumShProjection(rayDir, outColor.x, SHcoeffs);
    for (uint j = 0; j < 81; j++)
       gTcoeffsR[vertexID * 81 + j] += SHcoeffs[j];
       
    //for G part
    accumShProjection(rayDir, outColor.y, SHcoeffs);
    for (uint j = 0; j < 81; j++)
       gTcoeffsG[vertexID * 81 + j] += SHcoeffs[j];
       
    //for B part
    accumShProjection(rayDir, outColor.z, SHcoeffs);
    for (uint j = 0; j < 81; j++)
       gTcoeffsB[vertexID * 81 + j] += SHcoeffs[j];
}

#include "Utils/Math/MathConstants.slangh"
#include "Utils/HostDeviceShared.slangh"
#include "Experimental/Scene/Lights/EmissiveLightSamplerType.slangh"

import Scene.Scene;
import Scene.Raytracing;
import Utils.Math.MathHelpers;
import Utils.Helpers;
import Utils.Sampling.SampleGenerator;
import Scene.Shading;
//import Experimental.Scene.Material.TexLODHelpers;
//import Experimental.Scene.Material.TexLODTypes;
import Scene.HitInfo;
//import ShRecurrence;

import Experimental.Scene.Lights.EnvMapSampler;
import Experimental.Scene.Lights.EmissiveLightSampler;  
import Experimental.Scene.Material.MaterialShading;
import Experimental.Scene.Lights.LightHelpers;


static const uint kSamplesPerPixel = 1;
static const uint kLightSamplesPerVertex = 1;
static const uint kMaxBounces = 1;
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
static const uint kMaxLightSamplesPerVertex = 8;
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

static int gMapCount = MapCount;
//Texture2D<float4> gVertexOutgoingRadiance[50000];
Texture2D<float4> gVertexOutgoingRadiance[61];
shared Buffer<uint> gMeshToIndex;
struct RayData
{
    float L;
};

/** Returns the interpolated vertex attributes for a given triangle.
    \param[in] meshInstanceID The mesh instance ID.
    \param[in] triangleIndex Index of the triangle in the given mesh.
    \param[in] barycentrics Barycentric coordinates in the triangle.
    \param[out] Unpacked fetched vertices which can be used for further computations involving individual vertices.
    \return Interpolated vertex attributes.
*/
float3 getVertexColor(uint meshInstanceID, uint triangleIndex, float3 barycentrics,float3 rayDir)
{
    //const uint3 vtxIndices = gScene.getIndices(meshInstanceID, triangleIndex);
    float3 barycentric = float3(0.33, 0.33, 0.34);
    //VertexData v = gScene.getVertexData(meshInstanceID, triangleIndex, barycentrics);
    VertexData v = gScene.getVertexData(meshInstanceID, triangleIndex, barycentric);
    float3 temp1, temp2;
    uint index = (gMeshToIndex[meshInstanceID] + triangleIndex) / 10000;
    uint offset = (gMeshToIndex[meshInstanceID] + triangleIndex) % 10000;
    uint x_offset = (offset / 100) * 64;
    uint y_offset = (offset % 100) * 16;
    //return v.normalW;
    float Costheta = dot(v.normalW, -rayDir);
    float3 planePart = Costheta * v.normalW + rayDir;
    float3 planePart1 = normalize(planePart);
    float theta = dot(planePart, planePart);
    float Cosphi = dot(planePart1, v.tangentW.xyz);
    float phi = acos(Cosphi) / 2.0 * M_1_PI;
    //return float3(theta, phi, index);
    theta = theta * 16;
    phi = phi * 64;
    uint u0 = uint(theta);
    uint v0 = uint(phi);
    temp1 = gVertexOutgoingRadiance[index][uint2(y_offset + min(u0+1,15), x_offset + min(v0+1,63))].xyz * (phi - v0) + gVertexOutgoingRadiance[index][uint2(y_offset + min(u0+1,15), x_offset + v0)].xyz * (1.f + v0 - phi);
    temp2 = gVertexOutgoingRadiance[index][uint2(y_offset + u0, x_offset + min(v0+1,63))].xyz * (phi - v0) + gVertexOutgoingRadiance[index][uint2(y_offset + u0, x_offset + v0)].xyz * (1.f + v0 - phi);
    //return gVertexOutgoingRadiance[index][uint2(u0,v0)].xyz;
    float3 radiance_interpolate = temp2 * (theta - u0) + temp1 * (1.f + u0 - theta);

    //return float3(index);
    return radiance_interpolate;
}

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
    pdf = M_1_PI / 2;
    return float3(d, z);
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

cbuffer CB{
    uint gSpp;
    uint gVertexCount;
};

//TODO delete the pathTracerParams
struct PathTracerData
{
    // PathTracerParams        params;             ///< PathTracer shared parameters.
    EnvMapSampler           envMapSampler;      ///< Environment map sampler.
    EmissiveLightSampler    emissiveSampler;    ///< Emissive light sampler.
};

ParameterBlock<PathTracerData> gData;

//output channels
shared RWTexture2D<float4> gOutputColor;

/** Unpacked payload for scatter ray.
    L=Le+Lr
*/
struct ScatterRayData
{
    bool terminated;        ///< Set to true when path is terminated.
    uint pathLength;        ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.).

    float3 L;              ///< radiance .

    float3 origin;          ///< Next path segment origin (unless path is terminated).
    float3 direction;       ///< Next path segment direction (unless path is terminated).

    float3 thp;             ///< Current path throughput. This is updated at each path vertex.

    float3 normal;          ///< Shading normal at the ray origin. Only used if MIS is enabled.
    float pdf;              ///< Probability density function with respect to solid angle at the last path vertex. Only used if MIS is enabled.

    float4 shadowRay;       ///< Shadow ray direction (xyz) and distance (w). Only used if Lr > 0.

    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).

    /** Create ray payload with default parameters.
    */
    static ScatterRayData create(SampleGenerator sg)
    {
        ScatterRayData d;
        d.terminated = false;
        d.pathLength = 0;
        d.L = float3(0, 0, 0);
        d.origin = float3(0, 0, 0);
        d.direction = float3(0, 0, 0);
        d.thp = float3(1, 1, 1);
        d.normal = float3(0, 0, 0);
        d.pdf = 0.f;
        d.shadowRay = float4(0, 0, 0, 0);
        d.sg = sg;

        return d;
    }
};

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
        float misWeight = evalMIS(1, rayData.pdf, kLightSamplesPerVertex, lightPdf);

        float3 Le = pt.envMapSampler.eval(dir);
        //envlight have some problem (scene/me?)

        // Accumulate emitted radiance weighted by path throughput.
        rayData.L = rayData.thp * Le  * misWeight;
        

    }
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
    ShadingData sd = prepareShadingData(v, gScene.getMaterialID(hitParams.getGlobalHitID()),gScene.materials[gScene.getMaterialID(hitParams.getGlobalHitID())], gScene.materialResources[gScene.getMaterialID(hitParams.getGlobalHitID())], -rayDir, 0.f);

     // Compute tangent space if it is invalid.
    if (!(dot(sd.B, sd.B) > 0.f))   // Note: Comparison written so that NaNs trigger
    {
        sd.B = perp_stark(sd.N);
        sd.T = cross(sd.B, sd.N);
    }

    // interpolate radiance weighted by barycentrics.    
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    rayData.L = getVertexColor(hitParams.getGlobalHitID(), PrimitiveIndex(), barycentrics, rayDir);

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
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    //TODO:spp is 20，need to do some division

    float3 Color_all = float3(0, 0, 0);
    for(int i = 0;i < gSpp;i++){
        // Sample ray
//#if USE_DEPTH_OF_FIELD
//    SampleGenerator sg = SampleGenerator.create(launchIndex, gParams.frameCount);
//    RayDesc ray = gScene.camera.computeRayThinlens(launchIndex, launchDim, sampleNext2D(sg)).toRayDesc();
        //#else
        //自带jitter
        RayDesc ray = gScene.camera.computeRayPinhole(launchIndex, launchDim).toRayDesc();
//#endif

        // Trace ray
        ScatterRayData rayData;
        rayData.terminated = false;
        rayData.pathLength = 0;
        rayData.origin = ray.Origin;
        rayData.direction = ray.Direction;
        rayData.thp = float3(1, 1, 1);
        rayData.normal = float3(0, 0, 0);
        rayData.pdf = 1.f;
        rayData.L = float3(0, 0, 0);
        uint rayFlags = 0; // TODO: Set cull mode from the app
        TraceRay(gRtScene, rayFlags, 0xff /* instanceInclusionM ask */, 0 /* hitIdx */, hitProgramCount, 0 /* missIdx */, ray, rayData);
        Color_all += rayData.L;
    }
    Color_all /= gSpp;
    gOutputColor[launchIndex] = float4(Color_all, 1);
    //gOutputColor[launchIndex] = gVertexOutgoingRadiance[0][uint2(2,0)];
}

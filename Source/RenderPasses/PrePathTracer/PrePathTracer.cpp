/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
/***************************************************************************
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# * Neither the name of NVIDIA CORPORATION nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**************************************************************************/
#include "PrePathTracer.h"

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

using ChannelList = std::vector<ChannelDesc>;

namespace
{
    const char kShadingShader[] = "RenderPasses/PrePathTracer/PrePathTracer.rt.slang";
    const char kParameterBlockName[] = "gData";

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 164u;
    const uint32_t kMaxAttributesSizeBytes = 8u;
    const uint32_t kMaxRecursionDepth = 12u;//no for sure whether it should be 12

    const char* kOutDir = "outDir";
    const char* kOutFilename = "outFilename";

    const ChannelList kOutputChannels =
    {
        { "OutGoingRadiance", "gOutGoingRadiance", "OutGoing radiance (different angle from vertex)" },
        { "OutGoingDistance", "gOutGoingDistance", "OutGoing distance (different angle from vertex)" },
        { "OutGoingNormal", "gOutGoingNormal", "OutGoing normal (different angle from vertex)" },
    };
};

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary & lib)
{
    lib.registerClass("PrePathTracer", "precompute outgoing radiance ", PrePathTracer::create);
}

PrePathTracer::SharedPtr PrePathTracer::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new PrePathTracer());
    return pPass;
}

/* it missed? in new version?
bool PrtPrecompute::init(const Dictionary& dict)
{
    // Create program
    std::string shaderDir = std::string(PROJECT_DIR);
    mOutDir = shaderDir;

    // Parse dictionary parameters
    bool flag = parseDictionary(dict);
    if (!flag) throw std::exception("Failed to parse dictionary");

    // Create ray tracing program
    RtProgram::Desc progDesc;
    progDesc.addShaderLibrary(kShaderFile).setRayGen("rayGen");
    progDesc.addHitGroup(0, "scatterClosestHit", "scatterAnyHit").addMiss(0, "scatterMiss");
    progDesc.addHitGroup(1, "", "shadowAnyHit").addMiss(1, "shadowMiss");
    progDesc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
    mRaytrace.pProgram = RtProgram::create(progDesc, kMaxPayloadSizeBytes, kMaxAttributesSizeBytes);
    assert(mTracer.pProgram);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);
    assert(mpSampleGenerator);

    return true;
}*/

Dictionary PrePathTracer::getScriptingDictionary()
{
    Dictionary dict;
    //in whitted ray tracer ,should be following:  serializePass<false>(dict);
    dict[kOutDir] = mOutDir;
    dict[kOutFilename] = mOutFilename;
    return dict;
}

RenderPassReflection PrePathTracer::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    //without input pass
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

/*still no sure is it useful/should appear here
void PrePathTracer::compile(RenderContext* pContext, const CompileData& compileData)
{
    mFrameSize = compileData.defaultTexDims;
}*/

void PrePathTracer::preprocessScene()
{
    uint32_t numMeshInstance = mpScene->getMeshInstanceCount();
    //read from 
    for (uint32_t i = 0; i < numMeshInstance; i++)
    {
        const MeshInstanceData& inst = mpScene->getMeshInstance(i);
        const MeshDesc& mesh = mpScene->getMesh(inst.meshID);
        const auto& mat = mpScene->getMaterial(inst.materialID);

        mNumVertex += mesh.getTriangleCount();
        //if (mat->isEmissive())
        //    break;
        //printf("%d %d %d\n", i, inst.meshID, inst.materialID);
    }

    // Create VertexToMesh buffer
    mVertexToMeshData.clear();
    mVertexToMeshData.reserve(mNumVertex);
    /*
    mpTcoeffsR = Buffer::createStructured(sizeof(float), mNumVertex * 81);
    mpTcoeffsG = Buffer::createStructured(sizeof(float), mNumVertex * 81);
    mpTcoeffsB = Buffer::createStructured(sizeof(float), mNumVertex * 81);*/
    //there was a trick to re-sort the mesh
    for (uint32_t i = 0; i < numMeshInstance; i++)
    {
        const MeshInstanceData& inst = mpScene->getMeshInstance(i);
        const MeshDesc& mesh = mpScene->getMesh(inst.meshID);

        for (uint j = 0; j < mesh.getTriangleCount(); j++)
        {
            mVertexToMeshData.push_back(uint2(i, j));
            //printf("%d %d %d\n", inst.meshID, inst.vbOffset, inst.ibOffset);
            //mVertexToMeshData.push_back(uint2(i, j));
        }
    }
    
    /*
    for (uint32_t i = 0; i < cnt - 1; i++)
    {
        for (uint32_t j = i + 1; j < cnt; j++)
        {
            if (vertexID[i] > vertexID[j])
            {
                uint2 tmp = mVertexToMeshData[i];
                mVertexToMeshData[i] = mVertexToMeshData[j];
                mVertexToMeshData[j] = tmp;
                uint32_t  tmp1 = vertexID[i];
                vertexID[i] = vertexID[j];
                vertexID[j] = tmp1;
            }
        }
    }*/


    if (mpVertexToMesh) mpVertexToMesh = nullptr;
    mpVertexToMesh = Buffer::createStructured(sizeof(uint2), mVertexToMeshData.size());
    mpVertexToMesh->setBlob((void*)(mVertexToMeshData.data()), 0, mVertexToMeshData.size() * sizeof(uint2));

    // Create buffer
    // mpTcoeffs = TypedBuffer<float>::create(mNumVertex * 81);

    muSpp = 16;
    mvSpp = 64;
    //mSqrtSpp = 256;
    mVertexCount = 0;

    mOutDir = "D:/Falcor-4.2/Source/RenderPasses/PrePathTracer/";
    FILE* fp = fopen((mOutDir + "debug_scene_T.txt").c_str(), "w");
    fprintf(fp, "%d\n", mNumVertex);
    fclose(fp);
}

bool PrePathTracer::initLights(RenderContext* pRenderContext)
{
    // Clear lighting data for previous scene.
    mpEnvMapSampler = nullptr;
    mpEmissiveSampler = nullptr;
    mUseEmissiveLights = mUseEmissiveSampler = mUseAnalyticLights = mUseEnvLight = false;

    // If we have no scene, we're done.
    if (mpScene == nullptr) return true;

    // Create environment map sampler if scene uses an environment map.
    if (mpScene->getEnvMap())
    {
        mpEnvMapSampler = EnvMapSampler::create(pRenderContext, mpScene->getEnvMap());
    }

    return true;
}

bool PrePathTracer::updateLights(RenderContext* pRenderContext)
{
    // If no scene is loaded, we disable everything.
    if (!mpScene)
    {
        mUseAnalyticLights = false;
        mUseEnvLight = false;
        mUseEmissiveLights = mUseEmissiveSampler = false;
        mpEmissiveSampler = nullptr;
        return false;
    }

    // Configure light sampling.
    mUseAnalyticLights = mpScene->useAnalyticLights();
    mUseEnvLight = mpScene->useEnvLight() && mpEnvMapSampler != nullptr;

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    bool lightingChanged = false;
    if (!mpScene->useEmissiveLights())
    {
        mUseEmissiveLights = mUseEmissiveSampler = false;
        mpEmissiveSampler = nullptr;
    }
    else
    {
        mUseEmissiveLights = true;
        mUseEmissiveSampler = true;

        if (!mUseEmissiveSampler)
        {
            mpEmissiveSampler = nullptr;
        }
        else
        {
            // Create emissive light sampler if it doesn't already exist.
            if (mpEmissiveSampler == nullptr)
            {
                switch (mSelectedEmissiveSampler)
                {
                case EmissiveLightSamplerType::Uniform:
                    mUniformSamplerOptions = std::static_pointer_cast<EmissiveUniformSampler>(mpEmissiveSampler)->getOptions();
                    mpEmissiveSampler = EmissiveUniformSampler::create(pRenderContext, mpScene, mUniformSamplerOptions);
                    break;
                case EmissiveLightSamplerType::LightBVH:
                    mLightBVHSamplerOptions = std::static_pointer_cast<LightBVHSampler>(mpEmissiveSampler)->getOptions();
                    mpEmissiveSampler = LightBVHSampler::create(pRenderContext, mpScene, mLightBVHSamplerOptions);
                    break;
                default:
                    logError("Unknown emissive light sampler type");
                }
                if (!mpEmissiveSampler) throw std::exception("Failed to create emissive light sampler");

                // recreateVars(); // Trigger recreation of the program vars.
            }

            // Update the emissive sampler to the current frame.
            assert(mpEmissiveSampler);
            lightingChanged = mpEmissiveSampler->update(pRenderContext);
        }
    }

    return lightingChanged;
}

void PrePathTracer::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    assert(pScene);

    mpScene = pScene;

    if (!initLights(pRenderContext)) throw std::exception("Failed to initialize lights");
    updateLights(pRenderContext);
    // TODO: addDefine 包括什么：sceneDefine和generator
    // TODO: 同时要setScene

    std::string shaderDir = std::string(PROJECT_DIR);
    mOutDir = shaderDir;

    // Create ray tracing program
    RtProgram::Desc progDesc;
    progDesc.addShaderLibrary(kShadingShader).setRayGen("rayGen");
    progDesc.addHitGroup(0, "scatterClosestHit", "scatterAnyHit").addMiss(0, "scatterMiss");
    progDesc.addHitGroup(1, "", "shadowAnyHit").addMiss(1, "shadowMiss");
    progDesc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

    progDesc.addDefines(pScene->getSceneDefines());

    mRaytrace.pProgram = RtProgram::create(progDesc, kMaxPayloadSizeBytes, kMaxAttributesSizeBytes);
    assert(mRaytrace.pProgram);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);
    assert(mpSampleGenerator);

    //mRaytrace.pProgram->addDefines(mpSampleGenerator->getDefines());


    //mRaytrace.pVars = RtProgramVars::create(mRaytrace.pProgram, mpScene);

    mRaytrace.pProgram->setScene(mpScene);

    /*
    auto pGlobalVars = mRaytrace.pVars->getRootVar();
    bool success = mpSampleGenerator->setShaderData(pGlobalVars);
    if (!success) throw std::exception("Failed to bind sample generator");
    */
    preprocessScene();

    uint i = 1;
}


void PrePathTracer::setTracerData(const RenderData& renderData)
{
    auto pBlock = mRaytrace.pParameterBlock;
    assert(pBlock);

    // Bind emissive light sampler.
    if (mUseEmissiveSampler)
    {
        assert(mpEmissiveSampler);
        bool success = mpEmissiveSampler->setShaderData(pBlock["emissiveSampler"]);
        if (!success) throw std::exception("Failed to bind emissive light sampler");
    }
}

void PrePathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    updateLights(pRenderContext);

    const uint2 targetDim = uint2(muSpp, mvSpp);


    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData[it.name]->asTexture().get();
            if (pDst) pRenderContext->clearTexture(pDst);
        }
        return;
    }

    setStaticParams(mRaytrace.pProgram.get());
    /*
    if (mUseEmissiveSampler)
    {
        // Specialize program for the current emissive light sampler options.
        assert(mpEmissiveSampler);
        if (mpEmissiveSampler->prepareProgram(mRaytrace.pProgram.get())) mRaytrace.pVars = nullptr;
    }*/

    if (!mRaytrace.pVars) prepareVars();
    assert(mRaytrace.pVars);
    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.


    // Set shared data into parameter block.
    setTracerData(renderData);
    if (!mPrecomputeDone)
    {

        // stage 1 RIS+SHADOW+Temporal
        auto pGlobalVars = mRaytrace.pVars->getRootVar();

        pGlobalVars["CB"]["gVertexCount"] = mVertexCount; // 如果frameCount不够 就只用spatial
        pGlobalVars["CB"]["guSpp"] = muSpp;
        pGlobalVars["CB"]["gvSpp"] = mvSpp;

        pGlobalVars["gVertexToMesh"] = mpVertexToMesh;
        //pGlobalVars["gTcoeffsR"] = mpTcoeffsR;
        //pGlobalVars["gTcoeffsG"] = mpTcoeffsG;
        //pGlobalVars["gTcoeffsB"] = mpTcoeffsB;
        
        // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
        auto bind = [&](const ChannelDesc& desc)
        {
            if (!desc.texname.empty())
            {
                auto pGlobalVars = mRaytrace.pVars;
                pGlobalVars[desc.texname] = renderData[desc.name]->asTexture();
            }
        };
        for (auto channel : kOutputChannels) bind(channel);

        // gRtScene, gScene 变量会自动bind, scene->render也会自动bind
        // 但是对于compute pass，gScene还是要手动绑定
        mpScene->raytrace(pRenderContext, mRaytrace.pProgram.get(), mRaytrace.pVars, uint3(targetDim, 1));


        mVertexCount++;
        if (mVertexCount == mNumVertex)
        {
            mPrecomputeDone = true;

            
            //float normFactor = 1.f / (muSpp * mvSpp);
            //float* bufData = static_cast<float*>(mpTcoeffsR->getData());
            /*
            const float* coeffsRData = reinterpret_cast<const float*>(mpTcoeffsR->map(Buffer::MapType::Read));
            mpTcoeffsR->unmap();
            const float* coeffsGData = reinterpret_cast<const float*>(mpTcoeffsG->map(Buffer::MapType::Read));
            mpTcoeffsG->unmap();
            const float* coeffsBData = reinterpret_cast<const float*>(mpTcoeffsB->map(Buffer::MapType::Read));
            mpTcoeffsB->unmap();
            FILE* fp = fopen((mOutDir + "coeffs.txt").c_str(), "w");
            //the coefficient print format as follow,idx:vertexID,j:the number of the basic function
            for (uint32_t i = 0; i < mNumVertex; i++)
            {
                fprintf(fp, "%d ", i);
                uint32_t idx = i * 81;
                for (uint32_t j = 0; j < 81; j++)
                    fprintf(fp, "%.6f ", coeffsRData[idx + j]);
                for (uint32_t j = 0; j < 81; j++)
                    fprintf(fp, "%.6f ", coeffsGData[idx + j]);
                for (uint32_t j = 0; j < 81; j++)
                    fprintf(fp, "%.6f ", coeffsBData[idx + j]);
            }
            fclose(fp);
            printf("this pass is over");*/
        }
    }
}

void PrePathTracer::renderUI(Gui::Widgets& widget)
{
}

void PrePathTracer::setStaticParams(RtProgram* pProgram) const
{
    Program::DefineList defines;
    defines.add("USE_ANALYTIC_LIGHTS", mUseAnalyticLights ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", mUseEmissiveLights ? "1" : "0");
    defines.add("USE_ENV_LIGHT", mUseEnvLight ? "1" : "0");
    defines.add("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    pProgram->addDefines(defines);
}

void PrePathTracer::prepareVars()
{
    assert(mpScene);
    assert(mRaytrace.pProgram);

    // Configure program.
    mRaytrace.pProgram->addDefines(mpSampleGenerator->getDefines());

    // Create program variables for the current program/scene.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mRaytrace.pVars = RtProgramVars::create(mRaytrace.pProgram, mpScene);
    assert(mRaytrace.pVars);

    // Bind utility classes into shared data.
    auto pGlobalVars = mRaytrace.pVars->getRootVar();
    bool success = mpSampleGenerator->setShaderData(pGlobalVars);
    if (!success) throw std::exception("Failed to bind sample generator");

    // Create parameter block for shared data.
    ProgramReflection::SharedConstPtr pReflection = mRaytrace.pProgram->getReflector();
    ParameterBlockReflection::SharedConstPtr pBlockReflection = pReflection->getParameterBlock(kParameterBlockName);
    assert(pBlockReflection);
    mRaytrace.pParameterBlock = ParameterBlock::create(pBlockReflection);
    assert(mRaytrace.pParameterBlock);

    // Bind static resources to the parameter block here. No need to rebind them every frame if they don't change.
    // Bind the light probe if one is loaded.
    if (mpEnvMapSampler) mpEnvMapSampler->setShaderData(mRaytrace.pParameterBlock["envMapSampler"]);

    // Bind the parameter block to the global program variables.
    mRaytrace.pVars->setParameterBlock(kParameterBlockName, mRaytrace.pParameterBlock);
}

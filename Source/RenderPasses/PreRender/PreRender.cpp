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
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "PreRender.h"

 // Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

using ChannelList = std::vector<ChannelDesc>;

namespace
{
    const char kShadingShader[] = "RenderPasses/PreRender/PreRender.rt.slang";
    const char kParameterBlockName[] = "gData";

    const char kTextureLODMode[] = "mode";

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 164;
    const uint32_t kMaxAttributesSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;

    const ChannelList kOutputChannels =
    {
        { "color",          "gOutputColor",               "Output color (sum of direct and indirect)"                },
    };
};

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary & lib)
{
    lib.registerClass("PreRender", "Render Pass Template", PreRender::create);
}

PreRender::SharedPtr PreRender::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new PreRender);
    FILE* fp = fopen((pPass->mOutDir + "debug_scene_T.txt").c_str(), "r");
    int tmpVal;
    fscanf(fp, "%d", &tmpVal);
    //printf("\n\n\n%s\n", (pPass->mOutDir + "debug_scene_T.txt").c_str());
    //printf("%d", tmpVal);
    pPass->mMapCount = 61;
    //pPass->mMapCount = tmpVal;
    printf("%d\n", pPass->mMapCount);
    for (int i = 0; i < pPass->mMapCount / 2; i++) {
        std::string mImageNam = std::to_string(i);
        pPass->mpTex[i] = Texture::createFromFile(pPass->mPrefix+mImageNam+pPass->mSuffix, pPass->mGenerateMips, pPass->mLoadSRGB);
        printf("%d\n", i);
    }
    return pPass;
}


bool PreRender::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (mpScene && mpScene->onKeyEvent(keyEvent)) return true;
    return false;
}

void PreRender::preprocessScene()
{
    mSpp = 20;
    mVertexCount = 0;
    //mSqrtSpp = 256;
    for (int i = mMapCount / 2; i < mMapCount; i++) {
        std::string mImageNam = std::to_string(i);
        mpTex[i] = Texture::createFromFile(mPrefix + mImageNam + mSuffix, mGenerateMips, mLoadSRGB);
        printf("%d\n", i);
    }
    uint32_t numMeshInstance = mpScene->getMeshInstanceCount();
    mMeshToIndexData.clear();
    mMeshToIndexData.reserve(numMeshInstance);
    uint32_t cnt = 0;
    //there was a trick to re-sort the mesh
    for (uint32_t i = 0; i < numMeshInstance; i++)
    {
        const MeshInstanceData& inst = mpScene->getMeshInstance(i);
        const MeshDesc& mesh = mpScene->getMesh(inst.meshID);
        mMeshToIndexData.push_back(cnt);
        cnt = cnt + mesh.getTriangleCount();
    }
    if (mpMeshToIndex) mpMeshToIndex = nullptr;
    mpMeshToIndex = Buffer::createStructured(sizeof(uint), mMeshToIndexData.size());
    mpMeshToIndex->setBlob((void*)(mMeshToIndexData.data()), 0, mMeshToIndexData.size() * sizeof(uint));
}

Dictionary PreRender::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection PreRender::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    //reflector.addOutput("dst");
    //reflector.addInput("src");
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void PreRender::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    assert(pScene);

    mpScene = pScene;
    // TODO: addDefine 包括什么：sceneDefine和generator
    // TODO: 同时要setScene

    std::string shaderDir = std::string(PROJECT_DIR);

    // Create ray tracing program
    RtProgram::Desc progDesc;
    progDesc.addShaderLibrary(kShadingShader).setRayGen("rayGen");
    progDesc.addHitGroup(0, "scatterClosestHit", "scatterAnyHit").addMiss(0, "scatterMiss");
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
}

void PreRender::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const uint2 targetDim = uint2(1280, 720);
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

    if (!mRaytrace.pVars) prepareVars();
    assert(mRaytrace.pVars);
    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.


    // Set shared data into parameter block.
    setTracerData(renderData);
    // renderData holds the requested resources
    auto pGlobalVars = mRaytrace.pVars->getRootVar();

    pGlobalVars["CB"]["gVertexCount"] = mVertexCount; // 如果frameCount不够 就只用spatial
    pGlobalVars["CB"]["gSpp"] = mSpp;

    pGlobalVars["gMeshToIndex"] = mpMeshToIndex;
    for (int i = 0; i < mMapCount; i++)
    {
        pGlobalVars["gVertexOutgoingRadiance"][i] = mpTex[i];
    }
    // auto& pTexture = renderData["src"]->asTexture();
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            auto pGlobalVars = mRaytrace.pVars;
            pGlobalVars[desc.texname] = renderData[desc.name]->asTexture();
        }
    };
    for (auto channel : kOutputChannels) bind(channel);


    mpScene->raytrace(pRenderContext, mRaytrace.pProgram.get(), mRaytrace.pVars, uint3(targetDim, 1));

    //pRenderContext->blit(mpTex[0]->getSRV(), renderData["color"]->getRTV());
}

void PreRender::setStaticParams(RtProgram* pProgram) const
{
    Program::DefineList defines;
    defines.add("USE_ANALYTIC_LIGHTS", mUseAnalyticLights ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", mUseEmissiveLights ? "1" : "0");
    defines.add("USE_ENV_LIGHT", mUseEnvLight ? "1" : "0");
    defines.add("MapCount", std::to_string(mMapCount));
    defines.add("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    pProgram->addDefines(defines);
}

void PreRender::prepareVars()
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

void PreRender::renderUI(Gui::Widgets& widget)
{
}

void PreRender::setTracerData(const RenderData& renderData)
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

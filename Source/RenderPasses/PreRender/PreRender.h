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
#pragma once

#include <vector>
#include "Falcor.h"
#include "FalcorExperimental.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Experimental/Scene/Lights/LightBVHSampler.h"

using namespace Falcor;

class PreRender : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<PreRender>;

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual std::string getDesc() override { return "Insert pass description here"; }
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }

protected:

    void preprocessScene();

    void setStaticParams(RtProgram* pProgram) const;

    PreRender() = default;

    Texture::SharedPtr mpTex[61];//

    virtual void recreateVars() {}

    void setTracerData(const RenderData& renderData);

    void prepareVars();

    bool mPrecomputeDone = false;

    // Internal state
    SampleGenerator::SharedPtr      mpSampleGenerator;
    Scene::SharedPtr                mpScene;

    EmissiveLightSampler::SharedPtr     mpEmissiveSampler;              ///< Emissive light sampler or nullptr if disabled.
    EnvMapSampler::SharedPtr            mpEnvMapSampler;                ///< Environment map sampler or nullptr if disabled.

    bool    mUseAnalyticLights = false;     ///< True if analytic lights should be used for the current frame. See compile-time constant in
    bool    mUseEnvLight = false;           ///< True if env map light should be used for the current frame. See compile-time constant in 
    bool    mUseEmissiveLights = false;     ///< True if emissive lights should be taken into account. See compile-time constant in
    bool    mUseEmissiveSampler = false;    ///< True if emissive light sampler should be used for the current frame. See compile-time constant in 

    std::string mOutDir = "D:/Falcor-4.2/Source/RenderPasses/PrePathTracer/";
    std::string mPrefix = "D:/temp_reserve/merge-test/Mogwai.PrePathTracer.OutGoingRadiance.";
    std::string mSuffix = ".exr";

    bool mGenerateMips = false;
    bool mLoadSRGB = false;
    // PRT viewer data=

    /*
    Texture::SharedPtr mpLegendre2345, mpLegendre6789;
    Sampler::SharedPtr mpLegendreSampler;
    */

    std::vector<uint> mMeshToIndexData;
    Buffer::SharedPtr mpMeshToIndex;
    uint mNumVertex;
    uint2 mFrameSize;
    uint mSpp;
    uint mVertexCount;
    int mMapCount;

    // Ray tracing resources
    struct
    {
        RtProgram::SharedPtr pProgram;
        RtProgramVars::SharedPtr pVars;
        ParameterBlock::SharedPtr pParameterBlock;
    } mRaytrace;
};

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
#pragma once

#include <vector>

#include "Falcor.h"
#include "FalcorExperimental.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Experimental/Scene/Lights/LightBVHSampler.h"
#include "RenderGraph/RenderPassHelpers.h"
//#include "Experimental/Scene/Material/TexLODTypes.slang"  // Using the enum with Mip0, RayCones, etc

#define SPP 4

using namespace Falcor;

class PrePathTracer : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<PrePathTracer>;

    //PrePathTracer();

    /** Create a new render pass object.
    \param[in] pRenderContext The render context.
    \param[in] dict Dictionary of serialized parameters.
    \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual std::string getDesc() override { return "precomputed outgoing radiance"; }
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    //virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override {}
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    //with no textureLod
    /*
    void setTexLODMode(const TexLODMode mode) { mTexLODMode = mode; }
    TexLODMode getTexLODMode() const { return mTexLODMode; }

    void setRayConeMode(const RayConeMode mode) { mRayConeMode = mode; }
    RayConeMode getRayConeMode() const { return mRayConeMode; }
    */

    //Compute

    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;


protected:
    //PrePathTracer(const Dictionary& dict);
    PrePathTracer() = default;

    void preprocessScene();

    virtual void recreateVars() {}

    void prepareVars();

    void setStaticParams(RtProgram* pProgram) const;

    bool initLights(RenderContext* pRenderContext);

    bool updateLights(RenderContext* pRenderContext);

    void setTracerData(const RenderData& renderData);

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
    ///< EmissiveLightSamplerType            mSelectedEmissiveSampler = EmissiveLightSamplerType::LightBVH;   Which emissive light sampler to use.
    EmissiveLightSamplerType            mSelectedEmissiveSampler = EmissiveLightSamplerType::Uniform;  ///< Which emissive light sampler to use.
    EmissiveUniformSampler::Options     mUniformSamplerOptions;         ///< Current options for the uniform sampler.
    LightBVHSampler::Options            mLightBVHSamplerOptions;        ///< Current options for the light BVH sampler.

    std::string mOutDir;
    std::string mOutFilename;

    uint32_t mNumVertex;
    uint2 mFrameSize;
    uint muSpp;
    uint mvSpp;
    uint32_t mVertexCount;

    std::vector<uint2> mVertexToMeshData;
    Buffer::SharedPtr mpVertexToMesh;

    //Buffer::SharedPtr mpTcoeffsR;//save the SH coeff R part
    //Buffer::SharedPtr mpTcoeffsG;//save the SH coeff G part
    //Buffer::SharedPtr mpTcoeffsB;//save the SH coeff B part


    // Ray tracing resources
    struct
    {
        RtProgram::SharedPtr pProgram;
        RtProgramVars::SharedPtr pVars;
        ParameterBlock::SharedPtr pParameterBlock;
    } mRaytrace;

};

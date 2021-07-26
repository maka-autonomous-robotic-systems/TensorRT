/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleReporting.h"

using namespace nvinfer1;
using namespace sample;

int main(int argc, char** argv)
{
    sample::setReportableSeverity(ILogger::Severity::kERROR);

    Arguments args = argsToArgumentsMap(argc, argv);
    AllOptions options;

    if (parseHelp(args))
    {
        AllOptions::help(std::cout);
        return EXIT_SUCCESS;
    }

    if (!args.empty())
    {
        bool failed{false};
        try
        {
            options.parse(args);
        }
        catch (const std::invalid_argument& arg)
        {
            sample::gLogError << arg.what() << std::endl;
            failed = true;
        }

        if (failed)
        {
            AllOptions::help(std::cout);
            return 1;
        }
    }
    else
    {
        options.helps = true;
    }

    if (options.helps)
    {
        AllOptions::help(std::cout);
        return 0;
    }

    setCudaDevice(options.system.device, sample::gLogVerbose);
    sample::gLogVerbose << std::endl;

    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    for (const auto& pluginPath : options.system.plugins)
    {
        sample::gLogVerbose << "Loading supplied plugin library: " << pluginPath << std::endl;
        samplesCommon::loadLibrary(pluginPath);
    }

    if (args.find("loadTwo") != args.end())
    {
        const int first_device = 1;
        const int second_device = 2;
        const std::string original_name = options.build.engine;
        auto launch_mode = args.find("parallel") != args.end() ? std::launch::async : std::launch::deferred;

        std::vector<std::future<void>> futures;
        futures.push_back(std::async(launch_mode,
            [&]()
            {
                cudaSetDevice(first_device);
                options.build.engine
                    = original_name.substr(0, original_name.size() - 4) + "_" + std::to_string(first_device) + ".trt";
                getEngine(options.model, options.build, options.system, sample::gLogVerbose);
            }));
        futures.push_back(std::async(launch_mode,
            [&]()
            {
                cudaSetDevice(second_device);
                options.build.engine
                    = original_name.substr(0, original_name.size() - 4) + "_" + std::to_string(second_device) + ".trt";
                getEngine(options.model, options.build, options.system, sample::gLogVerbose);
            }));

        for (auto& future : futures)
        {
            future.get();
        }
    }
    else if (options.build.load)
    {
        options.build.engine = options.build.engine.substr(0, options.build.engine.size() - 4) + "_"
            + std::to_string(options.system.device) + ".trt";
        getEngine(options.model, options.build, options.system, sample::gLogError);
    }
    else
    {
        options.build.engine = options.build.engine.substr(0, options.build.engine.size() - 4) + "_"
            + std::to_string(options.system.device) + ".trt";
        getEngine(options.model, options.build, options.system, sample::gLogError);
    }
    return 0;
}

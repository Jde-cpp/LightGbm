#export CXX=g++-8
cmake_minimum_required(VERSION 3.16)
project(Jde.Dts.Lgb)
set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../Framework/cmake )
include(SetUpConfigurations)
#Custom######################################################
include_directories( "/home/duffyj/code/libraries/LightGBM/include" )
include_directories( "/home/duffyj/code/libraries/eigen" )
#############################################################
add_library( Jde.Dts.Lgb SHARED LightGbm.cpp Booster.cpp Dataset.cpp BoosterParams.cpp )
#############################################################
add_compile_definitions(THREAD_LOCAL=thread_local)
target_precompile_headers(Jde.Dts.Lgb
  PRIVATE
	<fstream>
	<boost/crc.hpp> 
	<boost/system/error_code.hpp>
	<spdlog/spdlog.h>
	<spdlog/sinks/basic_file_sink.h>
	<spdlog/fmt/ostr.h>
	<LightGBM/c_api.h>
	<LightGBM/dataset.h>
	<Eigen/Dense>
	<Eigen/Sparse>
	<nlohmann/json.hpp>
	"../../Framework/source/TypeDefs.h"
	"../../Eigen/source/EMatrix.h"
	"../../Framework/source/log/Logging.h"
	"../../Framework/source/JdeAssert.h"
	"../../Framework/source/StringUtilities.h"
	"../../Framework/source/threading/Pool.h"
	"../../Framework/source/math/MathUtilities.h"
	"../../Framework/source/io/File.h"
)
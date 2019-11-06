#include <fstream>
#ifndef _MSC_VER
	#define THREAD_LOCAL thread_local
#endif

#pragma warning( disable : 4245) 
#include <boost/crc.hpp> 
#pragma warning( default : 4245) 
#include <boost/system/error_code.hpp>
#ifndef __INTELLISENSE__
	#include <spdlog/spdlog.h>
	#include <spdlog/sinks/basic_file_sink.h>
	#include <spdlog/fmt/ostr.h>
#endif


#include <LightGBM/c_api.h>
#include <LightGBM/dataset.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nlohmann/json.hpp>

#include "../../Framework/source/TypeDefs.h"
#include "../../Eigen/source/EMatrix.h"

#pragma warning (disable:4275)
#pragma warning (disable:4251)
//#include "../dts/TypeDefs.h"
#include "../../Framework/source/log/Logging.h"
#include "../../Framework/source/JdeAssert.h"
#include "../../Framework/source/StringUtilities.h"
#include "../../Framework/source/threading/Pool.h"
#include "../../Framework/source/math/MathUtilities.h"
#include "../../Framework/source/io/File.h"


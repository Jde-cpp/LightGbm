#include "BoosterParams.h"
#define var const auto

namespace Jde::AI::Dts::LightGbm
{
#pragma region Constructors
	BoosterParams::BoosterParams()noexcept:
		IBoosterParams( DoubleParams, StringParams, UIntParams )
	{
		if( !DoubleParams.size() )
		{
			std::for_each( IBoosterParams::DefaultDoubleParams.begin(), IBoosterParams::DefaultDoubleParams.end(), [](var& value){ DoubleParams.emplace(value); } );
			std::for_each( IBoosterParams::DefaultStringParams.begin(), IBoosterParams::DefaultStringParams.end(), [](var& value){ StringParams.emplace(value); } );
			std::for_each( IBoosterParams::DefaultUIntParams.begin(), IBoosterParams::DefaultUIntParams.end(), [](var& value){ UIntParams.emplace(value); } );

			UIntParams.emplace( Leaves.Name );
			DoubleParams.emplace( BaggingFraction.Name );
			UIntParams.emplace( BaggingFrequency.Name );
			UIntParams.emplace( MaxBin.Name );
			UIntParams.emplace( MinSumHessianInLeaf.Name );
			UIntParams.emplace( "num_threads" );

			DoubleParams.emplace( FeatureFraction.Name );
			StringParams.emplace( Device.Name );
			StringParams.emplace( BoostingType.Name );
			StringParams.emplace( Task.Name );
			StringParams.emplace( Metric.Name );
			StringParams.emplace( Verbose.Name );
			_parameters.emplace( ThreadParamName(), make_shared<TParameter<uint>>(ThreadCount) );
		}
			_parameters.emplace( Leaves.Name, make_shared<TParameter<uint>>(Leaves) );
			_parameters.emplace( BaggingFraction.Name, make_shared<TParameter<double>>(BaggingFraction) );
			_parameters.emplace( BaggingFrequency.Name, make_shared<TParameter<uint>>(BaggingFrequency) );
			_parameters.emplace( MaxBin.Name, make_shared<TParameter<uint>>(MaxBin) );
			_parameters.emplace( MinSumHessianInLeaf.Name, make_shared<TParameter<uint>>(MinSumHessianInLeaf) );
			_parameters.emplace( FeatureFraction.Name, make_shared<TParameter<double>>(FeatureFraction) );
			_parameters.emplace( Device.Name, make_shared<TParameter<string>>(Device) );
			_parameters.emplace( BoostingType.Name, make_shared<TParameter<string>>(BoostingType) );
			_parameters.emplace( Task.Name, make_shared<TParameter<string>>(Task) );
			_parameters.emplace( Metric.Name, make_shared<TParameter<string>>(Metric) );
			_parameters.emplace( Verbose.Name, make_shared<TParameter<string>>(Verbose) );
			_parameters.emplace( ThreadParamName(), make_shared<TParameter<uint>>(ThreadCount) );
			_parameters.emplace( Objective.Name, make_shared<TParameter<string>>(Objective) );
	}

	const TParameter<uint> BoosterParams::Leaves = TParameter<uint>( "num_leaves", 30, 2 );						//range(12,257,100) #number of leaves in one treesp default=31
	const TParameter<double> BoosterParams::FeatureFraction = TParameter<double>( "feature_fraction", .36, .0, false);			//doesn't matter  #np.arange(.87,.90,.01) #default=1.0  Random select part of features on each iteration if feature_fraction smaller than 1.0. For example, if set to 0.8, will select 80% features before training each tree.
	const TParameter<double> BoosterParams::BaggingFraction = TParameter<double>( "bagging_fraction", .81, .01);			//np.arange(.80,.88,.01)#np.arange(.86,.89,.01) #default=1.0,  Like feature_fraction, but this will random select part of data
	const TParameter<uint> BoosterParams::BaggingFrequency = TParameter<uint>( "bagging_freq", 2, 1, true);					//Frequency for bagging, 0 means disable bagging. k means will perform bagging at every k iteration.
	const TParameter<uint> BoosterParams::MaxBin = TParameter<uint>( "max_bin", 128, 16 );
	const TParameter<uint> BoosterParams::MinSumHessianInLeaf = TParameter<uint>( "min_sum_hessian_in_leaf", 20, 1, false );
	const TParameter<string> BoosterParams::Task = TParameter<string>( "task", "train" );
	const TParameter<string> BoosterParams::BoostingType = TParameter<string>( "boosting_type", "gbdt" );
	const TParameter<string> BoosterParams::Metric = TParameter<string>( "metric", "l2" );
	const TParameter<string> BoosterParams::Verbose = TParameter<string>( "verbose", "0" );
	const TParameter<string> BoosterParams::Device = TParameter<string>( "device", "gpu" );


	BoosterParams::BoosterParams( const fs::path& paramFile )noexcept(false):
		BoosterParams()
	{
		if( !fs::exists(paramFile) )
			THROW( LogicException( fmt::format("{} does not exist.", paramFile.string())) );

		std::ifstream is( paramFile );
		//is.exceptions( std::ios::failbit | std::ios::badbit );
		//is.open( filename.c_str() );
		if( !is.good() )
			THROW( LogicException( fmt::format("{} could not read.", paramFile.string())) );
		Read( is );
	}
	BoosterParams::BoosterParams( std::istream& is )noexcept:
		BoosterParams()
	{
		Read( is );
	}
#pragma endregion
#pragma region BoosterParams
	uint BoosterParams::NumberOfLeavesValue()const noexcept
	{
		var pParam = std::dynamic_pointer_cast<TParameter<uint>>( (*this)["num_leaves"] );
		return pParam->Initial;
	}
	string BoosterParams::GetMetric()const noexcept
	{
		var pMetric = std::dynamic_pointer_cast<TParameter<string>>( (*this)["metric"] );
		return pMetric->Initial;
	}
	void BoosterParams::SetMetric( string_view metric )noexcept
	{
		var pMetric = std::dynamic_pointer_cast<TParameter<string>>( (*this)["metric"] );
		auto objective=""sv;
		if( metric=="l2" )
			objective = "regression_l2";
		else if( metric=="l1" )
			objective = "regression_l1";
		else if( metric=="quantile" || metric=="huber" || metric=="fair" || metric=="poisson" || metric=="gamma" || metric=="mape" || metric=="tweedie" )
			objective = metric;
		else
		{
			objective = metric;  
			if( metric!="regression" )
				WARN( "Unknown metric '{}'", metric );
		}
 		pMetric->Initial = metric;
		var pObjective = std::dynamic_pointer_cast<TParameter<string>>( (*this)["objective"] );
		pObjective->Initial = objective;
	}

	string BoosterParams::DeviceValue()const noexcept
	{
		var pDevice = std::dynamic_pointer_cast<TParameter<string>>( (*this)["device"] );
		return pDevice->Initial;
	}
	void BoosterParams::SetCpu()const noexcept
	{
		auto pDevice = std::dynamic_pointer_cast<TParameter<string>>( (*this)["device"] );
		pDevice->Initial = "cpu";
	}
	void BoosterParams::SetGpu()const noexcept
	{
		auto pDevice = std::dynamic_pointer_cast<TParameter<string>>( (*this)["device"] );
		pDevice->Initial = "gpu";
	}

	set<string> BoosterParams::DoubleParams;
	set<string> BoosterParams::StringParams;
	set<string> BoosterParams::UIntParams;
#pragma endregion
}
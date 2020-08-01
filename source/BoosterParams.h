#pragma once
#include <list>
#include <sstream>
//#include <nlohmann/json.hpp>
#include "../../DecisionTree/source/IBoosterParams.h"
#include "Exports.h"

namespace Jde::AI::Dts::LightGbm
{
#pragma region BoosterParams
	struct JDE_GBM_VISIBILITY BoosterParams : public Dts::IBoosterParams
	{
		IBoosterParamsPtr Create()const noexcept override{ return make_shared<BoosterParams>(); }
		IBoosterParamsPtr Clone()const noexcept override{ return make_shared<BoosterParams>(*this); }

		BoosterParams()noexcept;
		BoosterParams( std::istream& is )noexcept;
		BoosterParams( const fs::path& path )noexcept(false);
		BoosterParams( const BoosterParams& )=default;
		virtual ~BoosterParams()=default;
		uint NumberOfLeavesValue()const noexcept;
		string GetMetric()const noexcept override;void SetMetric( string_view metric )noexcept override;
		string DeviceValue()const noexcept override; void SetCpu()const noexcept  override; void SetGpu()const noexcept  override;/*const because not significant*/
		string_view ThreadParamName()const noexcept override{return "num_threads"sv;}
	private:
		const static TParameter<double> BaggingFraction;			//np.arange(.80,.88,.01)#np.arange(.86,.89,.01) #default=1.0,  Like feature_fraction, but this will random select part of data
		const static TParameter<double> FeatureFraction;			//doesn't matter  #np.arange(.87,.90,.01) #default=1.0  Random select part of features on each iteration if feature_fraction smaller than 1.0. For example, if set to 0.8, will select 80% features before training each tree.

		const static TParameter<uint> BaggingFrequency;				//Frequency for bagging, 0 means disable bagging. k means will perform bagging at every k iteration.
		const static TParameter<uint> MaxBin;
		const static TParameter<uint> MinSumHessianInLeaf;
		const static TParameter<uint> Leaves;						//range(12,257,100) #number of leaves in one treesp default=31

		const static TParameter<string> BoostingType;
		const static TParameter<string> Device;
		const static TParameter<string> Metric;
		const static TParameter<string> Task;
		const static TParameter<string> Verbose;

		static set<string> DoubleParams;
		static set<string> StringParams;
		static set<string> UIntParams;

		//string to_string()const noexcept override;
		//void Read( std::istream& is )noexcept;
		//nlohmann::json ToJson()const noexcept;
	};
#pragma endregion
}
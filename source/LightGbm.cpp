#pragma region Includes
#include "LightGbm.h"
#include <sstream>
#include <list>
//#include <boost/range/combine.hpp>
#include "BoosterParams.h"
#include "Booster.h"
#include "Dataset.h"
#define var const auto

#pragma endregion Includes

Jde::AI::Dts::IDecisionTree* GetDecisionTree()
{
	return new Jde::AI::Dts::LightGbm::DecisionTree();
}

namespace Jde::AI::Dts::LightGbm
{
#pragma region Defines

	using Eigen::MatrixXf;
	using Eigen::VectorXf;
	sp<IBooster> DecisionTree::CreateBooster( const fs::path& path )const noexcept(false)
	{
		return make_shared<Booster>( IO::FileUtilities::ToString(path) );
	}
	IBoosterParamsPtr DecisionTree::LoadParams( const fs::path& file )const noexcept(false)
	{
		ifstream is(file);
		if( !is.good() )
			THROW( Exception( fmt::format("Could not open file {}", file.string()) ) );
		return make_shared<AI::Dts::LightGbm::BoosterParams>( is );
	}

	IBoosterParamsPtr DecisionTree::LoadDefaultParams(string_view /*objective*/)const noexcept(false)
	{
		return make_shared<AI::Dts::LightGbm::BoosterParams>();
	}

	sp<IBooster> DecisionTree::CreateBooster( const IBoosterParams& params, sp<const IDataset>& train, sp<const IDataset> pValidation )
	{
		return make_shared<Booster>( params, train, pValidation );
	}

	sp<IDataset> DecisionTree::CreateDataset( const Eigen::MatrixXf& matrix, const Eigen::VectorXf& y, const IBoosterParams* pParams, const std::vector<string>* pColumnNames/*=nullptr*/, shared_ptr<const IDataset> pTrainingDataset )
	{
		return make_shared<Dataset>( matrix, y, pParams, pColumnNames, pTrainingDataset ? pTrainingDataset.get() : nullptr );
	}


#pragma endregion
}	
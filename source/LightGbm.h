#pragma once
#include "Exports.h"
#include "../../DecisionTree/source/IBoosterParams.h"
#include "../../DecisionTree/source/IDecisionTree.h"


//namespace{ namespace Eigen{class MatrixXf;} }
extern "C" JDE_GBM_VISIBILITY Jde::AI::Dts::IDecisionTree* GetDecisionTree();
namespace Jde::AI::Dts
{
	struct IBoosterParams; 
	struct IBooster;
namespace LightGbm
{
	struct BoosterParams;
	struct Booster;
	struct JDE_GBM_VISIBILITY DecisionTree : public IDecisionTree
	{
		DecisionTree():IDecisionTree{"lgb"}{}
		sp<IBooster> CreateBooster( const fs::path& file )const noexcept(false)override;
		sp<IBooster> CreateBooster(  const IBoosterParams& params, sp<const IDataset>& train, sp<const IDataset> pValidation )override;
		//fs::path BaseDir()const noexcept override{return _baseDir; }
		IBoosterParamsPtr LoadParams( const fs::path& file )const noexcept(false) override;
		IBoosterParamsPtr LoadDefaultParams( string_view objective )const noexcept(false) override;
		//sp<Jde::AI::Dts::IBooster> Train( const Eigen::MatrixXf& x, const Eigen::VectorXf& y, const IBoosterParams& params, uint count, const std::vector<string>& columnNames )noexcept(false)override;
		sp<IDataset> CreateDataset( const Eigen::MatrixXf& matrix, const Eigen::VectorXf& y, const IBoosterParams* pParams, const std::vector<string>* pColumnNames/*=nullptr*/, sp<const IDataset> pTrainingDataset ) override;
		string_view DefaultRegression()const noexcept override{return "l2"sv;}
	};

	//	LightGbm::Train(Eigen::Matrix<float, -1, -1, 0, -1, -1> const&, Eigen::Matrix<float, -1, 1, 0, -1, 1> const&, Jde::AI::LightGbm::BoosterParams const&, unsigned long, std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&)
	//JDE_GBM_VISIBILITY sp<IBooster> Train( const Eigen::MatrixXf& x, const Eigen::VectorXf& y, const AI::LightGbm::BoosterParams& params, uint count, const std::vector<string>& columnNames )noexcept(false);
	//JDE_GBM_VISIBILITY IBoosterParamsPtr Tune( vector<unique_ptr<Eigen::MatrixXf>>& xs, vector<Math::VPtr<>>& ys, uint testCount, const fs::path& saveStem, uint foldCount, const fs::path& paramStart )noexcept(false);
	//JDE_GBM_VISIBILITY IBoosterParamsPtr TuneOne( vector<unique_ptr<Eigen::MatrixXf>>& xs, vector<Math::VPtr<>>& ys, const fs::path& saveStem, uint foldCount, string_view parameterName )noexcept(false);
	//float Evaluate( BoosterTypes::Contract& )
//Testing:
	//std::tuple<uint,vector<double>> CrossValidate( const IBoosterParams& parameters, Eigen::MatrixXf& x, Math::Vector<>& y, uint foldCount, uint trainingRounds, string_view logSuffix  )noexcept(false);
}}
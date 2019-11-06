#pragma once
#include "../../DecisionTree/source/IDataset.h"
#include "Exports.h"

namespace Jde::AI::Dts
{
	struct IBoosterParams;
namespace LightGbm
{
	typedef LightGBM::Dataset* HDataset;

	class JDE_GBM_VISIBILITY Dataset : public IDataset
	{
	public:
		Dataset( const Eigen::MatrixXf& matrix, const IBoosterParams* pParams, const std::vector<string>* pColumnNames, const IDataset* pTrainingDataset )noexcept(false);
		Dataset( const Eigen::MatrixXf& matrix, const Eigen::VectorXf& y, const IBoosterParams* pParams, const std::vector<string>* pColumnNames/*=nullptr*/, const IDataset* pTrainingDataset )noexcept(false);

		~Dataset();
		const HDataset Handle()const{return _handle;}

	private:
		HDataset _handle;
		uint _maxFeatureLength{1024};
	};
}}
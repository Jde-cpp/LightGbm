#pragma once
#include "Exports.h"
#include "../../Eigen/source/EMatrix.h"
#include "../../DecisionTree/source/IBooster.h"

namespace Jde::AI::Dts
{
	struct IBoosterParams;
	struct IDataset;
	enum class EFeatureImportance : uint8;

namespace LightGbm
{
	struct BoosterParams;
	typedef void* BoosterHandle;
	struct JDE_GBM_VISIBILITY Booster : public Jde::AI::Dts::IBooster
	{
		//Booster( const BoosterParams& params, const Dataset& ds )noexcept(false);
		Booster( const IBoosterParams& params, sp<const IDataset>& ds, sp<const IDataset> pValidation=nullptr )noexcept(false);
		Booster( const string& model )noexcept(false);
		Booster( const Booster& ) = delete;
		virtual ~Booster();

		Booster& operator=(const Booster&) = delete;

		bool UpdateOneIteration( int index=-1 )noexcept(false) override;
		sp<vector<double>> Predict( const Eigen::MatrixXf& matrix )noexcept(false);
		double Predict( const Math::RowVector<float,-1>& vector )noexcept(false)override;
		double Predict( const double* pFeatures )noexcept(false)override;
		uint GetEvaluationCounts()const noexcept(false);
		vector<double> GetEvaluation( bool validation, uint iteration=0 )const noexcept(false) override;

		MapPtr<string,double> FeatureImportances( EFeatureImportance eFeatureImportance )const noexcept(false) override;
		string to_string( uint iterationNumber = 0 )const noexcept(false) override;
		void LoadModelFromString( string_view model )noexcept(false);
		uint FeatureCount()const noexcept(false);
		static uint SaveIfElse( const fs::path& modelPath, string_view namespaceName, ostream& osCpp )noexcept(false);
		static uint ModelCount( const fs::path& modelPath )noexcept(false);
		const vector<string>& FeatureNames()const noexcept override;
		const uint TrainRowCount;
		const uint ValidationRowCount;
		double BestScore{ std::numeric_limits<double>::max() };
		constexpr static char FileSuffix[] = ".model";
		void LoadBestIteration()noexcept(false)override;
		void Save( const fs::path& path )noexcept(false) override;
	private:
		void Release()noexcept;

		sp<vector<double>> FeatureImportanceValues( EFeatureImportance eFeatureImportance )const;
		mutable vector<string> _featureNames;
		BoosterHandle _handle{ nullptr };
		static uint _gpuMaxBinSize;
	};
	typedef sp<Booster> BoosterPtr;
}}
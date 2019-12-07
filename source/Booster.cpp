#include "Booster.h"
#include "Dataset.h"
#include "BoosterParams.h"
#include <LightGBM/boosting.h>
#define var const auto


namespace Jde::AI::Dts::LightGbm
{
	uint Booster::_gpuMaxBinSize=std::numeric_limits<uint>::max();
	//static char* Booster::FileSuffix = ".model";
	Booster::Booster( const IBoosterParams& params, sp<const IDataset>& pTraining, sp<const IDataset> pValidation )noexcept(false):
		TrainRowCount{ pTraining->RowCount },
		ValidationRowCount{ pValidation ? pValidation->RowCount : 0 }
	{
		if( params.MaxBinValue()>=_gpuMaxBinSize )
			params.SetCpu();
		auto parameterString = params.to_string();
		auto pLgb = dynamic_pointer_cast<const Dataset>( pTraining );
		var failed = LGBM_BoosterCreate( pLgb->Handle(), parameterString.c_str(), &_handle );
		if( failed )
		{
			var message = LastErrorMsg();
			GetDefaultLogger()->error( message );
			if( params.DeviceValue()=="gpu" )  //&& message==fmt::format("bin size {} cannot run on GPU", params.MaxBinValue())
			{
				params.SetCpu();
				_gpuMaxBinSize = params.MaxBinValue();
				auto parameterString = params.to_string();
				var failed = LGBM_BoosterCreate( pLgb->Handle(), parameterString.c_str(), &_handle );
				if( failed )
					THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
			}
			else
				THROW( Exception(fmt::format("({}) - {}", failed, message)) );
		}
		if( pValidation!=nullptr )
		{
			auto pMyValidation = dynamic_pointer_cast<const Dataset>( pValidation );
			ASSRT_NN( _handle );
			var failed2 = LGBM_BoosterAddValidData( _handle, pMyValidation->Handle() );
			if( failed2 )
				THROW( Exception(fmt::format("({}) - {}", failed2, LastErrorMsg())) );
		}
	}
	Booster::Booster( const string& model )noexcept(false):
		TrainRowCount{0},
		ValidationRowCount{0}
	{
		LoadModelFromString( model );
	}

	void Booster::Save( const fs::path& path )noexcept(false)
	{
		IO::FileUtilities::Save( path, to_string() ); 
	}

	void Booster::LoadModelFromString( const string_view model )noexcept(false)
	{
		Release();
		int iterationCount;
		var failed = LGBM_BoosterLoadModelFromString( string(model).c_str(), &iterationCount, &_handle );
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
 	}
	bool Booster::UpdateOneIteration( int index )noexcept(false)
	{
		int isFinished;
		var failed = LGBM_BoosterUpdateOneIter( _handle, &isFinished );
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		return isFinished!=1;
	}
	MapPtr<string,double> Booster::FeatureImportances( EFeatureImportance eFeatureImportance )const noexcept(false)
	{
		var names = FeatureNames();
		var pImportance = FeatureImportanceValues( eFeatureImportance );
		ASSERT( names.size()==pImportance->size() );
		auto pResults = make_shared<map<string,double>>();
		for( uint i=0; i<names.size() && i<pImportance->size(); ++i )
			pResults->emplace( names[i], (*pImportance)[i] );
		return pResults;
	}
	sp<vector<double>> Booster::FeatureImportanceValues( EFeatureImportance eFeatureImportance )const
	{
		auto pResults = make_shared<vector<double>>( FeatureCount() );
		var failed = LGBM_BoosterFeatureImportance( _handle, -1, 1, pResults->data() );
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		return pResults;
	}

	uint Booster::FeatureCount()const noexcept(false)
	{
		int length;
		var failed = LGBM_BoosterGetNumFeature( _handle, &length );
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		return length;
	}

	const vector<string>& Booster::FeatureNames()const noexcept
	{
		if( !_featureNames.size() )
		{
			var featureCount = FeatureCount();
			auto pFeatures = new char*[featureCount];//TODO make this a string.
			for( uint i=0; i<featureCount; ++i )
				pFeatures[i] = new char[1024];
			int actualSize;
			var failed = LGBM_BoosterGetFeatureNames( _handle, &actualSize, pFeatures );
			_featureNames.reserve( featureCount );
			if( !failed )
			{
				for( uint i=0; i<featureCount; ++i )
					_featureNames.push_back( string(pFeatures[i]) );
			}
			for( uint i=0; i<featureCount; ++i )
				delete [] pFeatures[i];
			delete [] pFeatures;
			if( failed )
				THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		}
		return _featureNames;
	}
	
	Booster::~Booster()
	{
		Release();
	}
	void Booster::Release()noexcept
	{
		_featureNames.clear();
		if( _handle )
		{
			var failed = LGBM_BoosterFree( _handle );
			_handle = nullptr;
			if( failed )
				ERR( "({}) - {}", failed, LastErrorMsg() );
		}
	}
	string Booster::to_string( uint iterationNumber )const noexcept(false)
	{
		string value; 
		int64_t size = 10;
		do
		{
			value.resize( size );
#ifdef _MSC_VER
			auto failed = LGBM_BoosterSaveModelToString( _handle, static_cast<int>(iterationNumber), 1, value.capacity(), &size, &value[0] );
#else
			auto failed = LGBM_BoosterSaveModelToString( _handle, static_cast<int>(iterationNumber), value.capacity(), &size, &value[0] );
#endif
			if( failed )
				THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		}while( (uint)size>value.capacity() );
		value.resize( size );
		return value;
	}
	//you can train on mutiple objectives, aoc, rmse, etc.  #of objectives
	uint Booster::GetEvaluationCounts()const noexcept(false)
	{
		int outLength;
		var failed = LGBM_BoosterGetEvalCounts( _handle, &outLength );
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		return outLength;
	}

	vector<double> Booster::GetEvaluation( bool validation, uint iteration )const noexcept(false)
	{
		auto results = vector<double>( GetEvaluationCounts() );
		int outLength;
		var failed = LGBM_BoosterGetEval( _handle, validation ? 1 : 0, &outLength, results.data() );
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		ASSERT( results.size()==(uint)outLength );

		return results;
	}

	void Booster::LoadBestIteration()noexcept(false)
	{
		LoadModelFromString( to_string(BestIteration()) );
	}
/*	void* Booster::Model( int iterationNumber )
	{
		string value; value.resize( 1024 );
		int64_t outLength;
		auto failed = LGBM_BoosterSaveModelToString( _handle, iterationNumber, value.capacity(), &outLength, &value[0] );
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		if( outLength>bufferLength )
		{
			value.resize( outLength );
			auto failed = LGBM_BoosterSaveModelToString( _handle, iterationNumber, value.capacity(), &outLength, &value[0] );
			if( failed )
				THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		}
	}
	*/
	
	sp<vector<double>> Booster::Predict( const Eigen::MatrixXf& matrix )noexcept(false)
	{
		const char* pszParameter = "";
		auto pResults = make_shared<vector<double>>( matrix.rows() );
		int64_t length;
		var failed = LGBM_BoosterPredictForMat( _handle, matrix.data(), C_API_DTYPE_FLOAT32, static_cast<int>(matrix.rows()), static_cast<int>(matrix.cols()), matrix.IsRowMajor, C_API_PREDICT_NORMAL, -1/*num_iteration*/, pszParameter, &length, pResults->data() );
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		ASSERT( pResults->size()==(uint)length );
		return pResults;
		//Compressed Sparse Column (CSC) or Compressed Sparse Row (CSR) sparse matrix
		//LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForCSR( BoosterHandle handle, const void* indptr, int indptr_type, const int32_t* indices, const void* data, int data_type, int64_t nindptr, int64_t nelem, int64_t num_col, int predict_type, int num_iteration, const char* parameter, int64_t* out_len, double* out_result );
		//LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForCSC( BoosterHandle handle, const void* col_ptr, int col_ptr_type, const int32_t* indices,const void* data, int data_type, int64_t ncol_ptr,int64_t nelem, int64_t num_row, int predict_type, int num_iteration, const char* parameter, int64_t* out_len, double* out_result );
	}
	double Booster::Predict( const Math::RowVector<float,-1>& vector )noexcept(false)
	{
		double result;
		int64_t length; const char* pszParameter = "";
		var failed = LGBM_BoosterPredictForMat( _handle, vector.data(), C_API_DTYPE_FLOAT32, 1, static_cast<int>(vector.cols()), vector.IsRowMajor, C_API_PREDICT_NORMAL, -1/*num_iteration*/, pszParameter, &length, &result );
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		ASSERT( 1==length );
		return result;
	}
	double Booster::Predict( const double* pFeatures )noexcept(false)
	{
		double result;
		int64_t length; const char* pszParameter = "";
		var failed = LGBM_BoosterPredictForMat( _handle, pFeatures, C_API_DTYPE_FLOAT64, 1, (int32_t)FeatureCount(), 0, C_API_PREDICT_NORMAL, -1/*num_iteration*/, pszParameter, &length, &result );
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		ASSERT( 1==length );
		return result;
	}

	uint Booster::SaveIfElse( const fs::path& modelPath, string_view namespaceName, ostream& osCpp )noexcept(false)
	{
		throw Exception( "Not implemented" );
/*		LightGBM::Boosting* pBoosting;
		try
		{
			pBoosting = LightGBM::Boosting::CreateBoosting("gbdt", modelPath.string().c_str() );
		}
		catch( const std::runtime_error& e )
		{
			THROW( IOException( "Could not parse {} - ", modelPath.string(), e.what() ) );
		}
		//pBoosting->MyModelToIfElse( string(namespaceName), osCpp );
		return pBoosting->MaxFeatureIdx()+1;*/
	}
	uint Booster::ModelCount( const fs::path& modelPath )noexcept(false)
	{
		//LightGBM::Boosting* pBoosting;
		uint count = 0;
		try
		{
			auto pBoosting = LightGBM::Boosting::CreateBoosting("gbdt", modelPath.string().c_str() );
			count = pBoosting->NumberOfTotalModel();
		}
		catch( const std::runtime_error& e )
		{
			THROW( IOException( "Could not parse {} - ", modelPath.string(), e.what() ) );
		}
		return count;
	}
}
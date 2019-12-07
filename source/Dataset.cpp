#include "Dataset.h"
#include <list>
#include "BoosterParams.h"
#define var const auto

namespace Jde::AI::Dts::LightGbm
{
	Dataset::Dataset( const Eigen::MatrixXf& matrix, const IBoosterParams* pParams, const std::vector<string>* pColumnNames, const IDataset* pTrainingDataset )noexcept(false):
		IDataset{ (uint)matrix.rows(), (uint)matrix.cols() }
	{
		void* pHandle;
		const Dataset* pMyTrainingDataset = dynamic_cast<const Dataset*>( pTrainingDataset );
		var failed = LGBM_DatasetCreateFromMat( matrix.data(), C_API_DTYPE_FLOAT32, static_cast<int>(matrix.rows()), static_cast<int>(matrix.cols()), matrix.IsRowMajor, pParams ? pParams->to_string().c_str() : "", pMyTrainingDataset ? pMyTrainingDataset->Handle() : nullptr, &pHandle );
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
		_handle = static_cast<HDataset>( pHandle );
		if( pColumnNames )
		{
			ASSRT_EQ( (uint)matrix.cols(), pColumnNames->size() );
			const char** pFeatures = new const char*[pColumnNames->size()];
			uint i = 0;
			_maxFeatureLength = 0;
			std::list<string> spaceNames;
			for( var& columnName : *pColumnNames )
			{
				if( columnName.find(" ")!=string::npos )
				{
					var value = StringUtilities::Replace( columnName, " ", "_" ); spaceNames.push_back( value );
					pFeatures[i++] = value.c_str();
				}
				else
					pFeatures[i++] = columnName.c_str();
				_maxFeatureLength = std::max( columnName.size(), _maxFeatureLength );
			}
			ASSRT_LT( 1024, _maxFeatureLength );//will crash on getFeatureNames
			var failed2 = LGBM_DatasetSetFeatureNames( _handle, pFeatures, static_cast<int>(pColumnNames->size()) );
			delete [] pFeatures;
			if( failed2 )
				THROW( Exception(fmt::format("({}) - {}", failed2, LastErrorMsg())) );
		}
	}
	Dataset::Dataset( const Eigen::MatrixXf& matrix, const Eigen::VectorXf& y, const IBoosterParams* pParams, const std::vector<string>* pColumnNames, const IDataset* pTrainingDataset )noexcept(false):
		Dataset( matrix, pParams, pColumnNames, pTrainingDataset )
	{
		ASSRT_EQ( matrix.rows(), y.rows() );
		var failed = LGBM_DatasetSetField( _handle, "label", y.data(), static_cast<int>(y.rows()), C_API_DTYPE_FLOAT32 ); //  or C_API_DTYPE_INT32
		if( failed )
			THROW( Exception(fmt::format("({}) - {}", failed, LastErrorMsg())) );
	} 

	Dataset::~Dataset()
	{
		var failed = LGBM_DatasetFree( _handle );
		if( failed )
			ERR( "({}) - {}", failed, LastErrorMsg() );
	}
}
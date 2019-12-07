#include "AutoProgramming.h"
#include <fstream>
#include "Booster.h"
#include "../../framework/Defines.h"

namespace Jde::AI::LightGbm
{
	void AutoProgramming::StartHeader( ostream& os )noexcept
	{
		os << "#pragma once" << endl
			<< "#include \"../Trees.h\"" << endl << endl;
	}

	void AutoProgramming::StartSource( ostream& os )noexcept
	{
		os << "#include \"stdafx.h\"" << endl
			<< "#include \"Models.h\"" << endl << endl
			<< "namespace Jde::AI::LightGbm::Dts" << endl
			<< "{" << endl
			<< "\tstd::unique_ptr<std::map<std::string_view,ITree&>> pTrees;" << endl
			<< "\tvoid InitializeTrees();" << endl
			<< "\tITree* GetPrediction( string_view namespaceName )" << endl
			<< "\t{" << endl
			<< "\t\tif( !pTrees )" << endl
			<< "\t\tInitializeTrees();" << endl
			<< "\t\tauto pTree = pTrees->find( namespaceName );" << endl
			<< "\t\treturn pTree==pTrees->end() ? nullptr : &pTree->second;" << endl
			<< "\t}" << endl
			<< "\tvoid InitializeTrees()" << endl
			<< "\t{" << endl
			<< "\t\tauto p = make_unique<std::map<std::string_view,ITree&>>();" << endl;
	}
	void AutoProgramming::EndSource( ostream& os )noexcept
	{
		os << "\t\tpTrees = move(p);" << endl
			<< "\t}" << endl
			<< "}";
	}

	void AutoProgramming::Append( ostream& osHeader, string namespaceName, const fs::path& sourcePath, ostream& osCombinedSource, string_view key, const fs::path& modelPath )noexcept(false)
	{
		std::ofstream os( sourcePath ); ASSRT_TR( os.good() );
		var featureCount = AI::Dts::LightGbm::Booster::SaveIfElse( modelPath.string(), namespaceName, os );
		osHeader << "namespace " << namespaceName << endl
					<< "{" << endl
					<< "\tstruct Tree : public TreeBase<" << featureCount << ">" << endl
					<< "\t{" << endl
					//<< "\t\tstd::unique_ptr<ITree> CreateInstance()noexcept override{ return unique_ptr<Tree>( new Tree() ); };" << endl
					<< "\t\tdouble Predict( const double* pFeatures )noexcept override;" << endl
					<< "\tprivate:" << endl
					<< "\t\tTree()noexcept; friend ITree;" << endl
					<< "\t\tstatic std::unique_ptr<Tree> _pInstance;" << endl
					<< "\t};" << endl
					<< "}" << endl;
		osCombinedSource << "\t\tp->emplace( \"" << key << "\", ITree::GetInstance<"<< namespaceName.substr(24) << "::Tree>() );" << endl;
	}
}
#pragma endregion

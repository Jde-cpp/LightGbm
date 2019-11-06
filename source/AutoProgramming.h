#pragma once
//#include "../../framework/TypeDefs.h"

namespace Jde
{
	using std::ostream;
namespace AI::LightGbm::AutoProgramming
{
	void StartHeader( ostream& os )noexcept;
	void StartSource( ostream& os )noexcept;
	void EndSource( ostream& os )noexcept;
	void Append( ostream& osHeader, string namespaceName, const fs::path& sourcePath, ostream& osCombinedSource, string_view key, const fs::path& modelPath )noexcept(false);
}}
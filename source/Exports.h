#pragma once
#ifdef Jde_Dts_Lgb_EXPORTS
	#ifdef _MSC_VER 
		#define JDE_GBM_VISIBILITY __declspec( dllexport )
	#else
		#define JDE_GBM_VISIBILITY __attribute__((visibility("default")))
	#endif
#else 
	#ifdef _MSC_VER
		#define JDE_GBM_VISIBILITY __declspec( dllimport )
		#define _GLIBCXX_USE_NOEXCEPT noexcept
		#if NDEBUG
			#pragma comment(lib, "Jde.AI.LightGbm.lib")
		#else
			#pragma comment(lib, "Jde.AI.LightGbm.lib")
		#endif
	#else
		#define JDE_GBM_VISIBILITY
	#endif
#endif 

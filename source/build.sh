#!/bin/bash
debug=${1:-1}
clean=${2:-0}

cd "${0%/*}"
echo "Build LightGbm"
if [ $clean -eq 1 ]; then
	make -C. clean DEBUG=$debug
	if [ $debug -eq 1 ]
	then
		ccache g++-8 -c -g -pthread -fPIC -std=c++17 -Wall -fsanitize=address -fno-omit-frame-pointer -Wno-unknown-pragmas -DJDE_GBM_EXPORTS -O0 pc.h -o.obj/debug2/stdafx.h.gch -I/home/duffyj/code/libraries/spdlog/include  -I/home/duffyj/code/libraries/LightGBM/include -I/home/duffyj/code/libraries/eigen -I/home/duffyj/code/libraries/json/include -I$BOOST_ROOT
	else
		ccache g++-8 -c -g -pthread -fPIC -std=c++17 -Wall -Wno-unknown-pragmas -DJDE_GBM_EXPORTS -march=native -DNDEBUG -O3 pc.h -o.obj/release2/stdafx.h.gch -I/home/duffyj/code/libraries/spdlog/include  -I/home/duffyj/code/libraries/LightGBM/include -I/home/duffyj/code/libraries/eigen -I/home/duffyj/code/libraries/json/include -I$BOOST_ROOT
	fi
	if [ $? -eq 1 ]; then
		exit 1
	fi
fi 
make -C. -j DEBUG=$debug
cd -
exit $?
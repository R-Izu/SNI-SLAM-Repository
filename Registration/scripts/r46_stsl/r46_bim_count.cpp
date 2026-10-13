// R46 §13-2：BIM 側で STSL が取る角の数と、三角形の記述子（corner triplet）の数を数える（登録はしない）。
// STSL のコードは変えない。使い方：r46_bim_count <登録の config（STSL の configs/interval/ の下）>
// config の floor の BIM を読み（ハッシュ表は読まない）、公式の ConstructHashTable と同じ処理で角と三つ組を作って数える。
#include "utils/utils.h"
#include "utils/cfg.h"
#include "frontend/geometry/bim.h"
#include "global_definition/global_definition.h"
#include <iostream>
int main(int argc, char **argv) {
	if (argc < 2) { std::cerr << "usage: r46_bim_count <config>" << std::endl; return 2; }
	InitGLOG(argv[1]);
	cfg::readParameters(argv[1]);
	BIMManager bim;
	bim.LoadBim(false);
	bim.ConstructHashTable(cfg::augmented);
	std::cout << "{\"floor\":\"" << cfg::floor << "\",\"n_corners\":" << bim.GetCorners().size()
	          << ",\"n_triplets\":" << bim.GetGeomSets().size() << "}" << std::endl;
	return 0;
}

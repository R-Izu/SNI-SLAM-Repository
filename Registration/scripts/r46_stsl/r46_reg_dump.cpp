// R46：STSL の reg_bm の "benchmark" と同じ処理で登録し、各 submap の結果を 1 行の JSON で書き出す。
// STSL のコードは変えない（このファイルは STSL のリポジトリの外に置き、同じライブラリにリンクするだけ）。
// 追加で書き出すもの：推定した変換 tf（水平化した submap 座標 → BIM）、最適スコア、候補ごとの
// 検証スコア（Score − PunishScore）と、候補が STSL 自身の成功判定（3 m・5°）を満たすか、正解の検証スコア。
// 使い方：r46_reg_dump <config の絶対パス> <出力 .jsonl>
#include "utils/utils.h"
#include "utils/cfg.h"
#include "utils/analysis.h"
#include "frontend/geometry/corner.h"
#include "frontend/geometry/bim.h"
#include "frontend/geometry/submap.h"
#include "backend/reglib.h"
#include "global_definition/global_definition.h"
#include <chrono>
#include <fstream>
#include <iomanip>

namespace fs = std::filesystem;

static int countFiles(const fs::path &d) {
	int n = 0;
	if (fs::exists(d) && fs::is_directory(d))
		for (const auto &e: fs::directory_iterator(d)) if (fs::is_regular_file(e.status())) ++n;
	return n;
}

static void writeMat(std::ofstream &o, const Eigen::Matrix4d &T) {
	o << "[";
	for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) o << std::setprecision(17) << T(i, j) << ((i == 3 && j == 3) ? "" : ",");
	o << "]";
}

int main(int argc, char **argv) {
	if (argc < 3) { std::cerr << "usage: r46_reg_dump <config> <out.jsonl>" << std::endl; return 2; }
	std::string config_path = argv[1];
	InitGLOG(config_path);
	cfg::readParameters(config_path);
	auto t0 = std::chrono::steady_clock::now();
	BIMManager bim;
	bim.LoadBim();
	double bim_load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	std::string seqDir = getROOT() + "processed/" + cfg::seq + "/" + std::to_string(cfg::interval) + "m";
	int n = countFiles(seqDir + "/occupied_pcd_flatten");
	std::ofstream out(argv[2]);
	for (int i = 0; i < n; ++i) {
		SubmapManager submap(i);
		auto ts = std::chrono::steady_clock::now();
		submap.LoadSubmap();
		auto results = bimreg::GlobalRegistration(submap, bim);
		double reg_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
		Analysis analysis(submap, bim, results);
		bool ok = bimreg::evaluate(results.tf, submap);
		out << "{\"index\":" << i << ",\"bim_load_s\":" << bim_load_s << ",\"reg_s\":" << reg_s
		    << ",\"success_stsl\":" << (ok ? "true" : "false") << ",\"optimal_score\":" << results.OptimalScore
		    << ",\"n_corres\":" << results.geoCorresSize << ",\"n_corners\":" << submap.GetCorners().size()
		    << ",\"tf\":";
		writeMat(out, results.tf);
		out << ",\"gt_xyyaw\":[" << submap.gt_[0] << "," << submap.gt_[1] << "," << submap.gt_[2] << "]";
		Eigen::Matrix4d Tgt = xyyawToSe3(submap.gt_[0], submap.gt_[1], submap.gt_[2], true);
		out << ",\"gt_score\":" << (results.verification.Score(Tgt) - results.verification.PunishScore(Tgt));
		out << ",\"candidates\":[";
		for (size_t k = 0; k < results.candidates.size(); ++k) {
			const auto &tf = results.candidates[k];
			double sc = results.verification.Score(tf) - results.verification.PunishScore(tf);
			out << (k ? "," : "") << "{\"score\":" << sc << ",\"correct\":" << (bimreg::evaluate(tf, submap) ? "true" : "false") << "}";
		}
		out << "]}" << std::endl;
		std::cout << "submap " << i << " success " << ok << " candidates " << results.candidates.size() << std::endl;
	}
	return 0;
}

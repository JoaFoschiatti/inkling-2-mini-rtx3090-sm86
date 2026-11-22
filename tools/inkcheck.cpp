// inkcheck: dump raw logits (teacher forcing) and compare two dumps.
//   inkcheck dump --model M --text F --out O [--ctx N] [--prefill P] [--prefill-rows L] [--ubatch U] [--decode D] [--gen G]
//   inkcheck cmp A B
// dump: prefill P tokens in ubatches of U (keeps the logits of the last L), then decode D tokens one by one, then G greedy tokens.
// cmp:  top-1 agreement, max logit difference, KL divergence and paired log-likelihood of the real next token.
// Run each dump with LD_LIBRARY_PATH set to the build to test.
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct header {
    char    magic[4];
    int32_t n_vocab;
    int32_t n_prefill_rows;
    int32_t n_decode_rows;
    int32_t n_gen;
};

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int cmd_dump(int argc, char ** argv) {
    std::string model_path, text_path, out_path;
    int n_ctx = 4096, n_prefill = 0, n_prefill_rows = 0, n_ubatch = 512, n_decode = 256, n_gen = 0, decode_batch = 1;
    bool kv_q8 = false;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--model")        model_path = next();
        else if (a == "--text")         text_path = next();
        else if (a == "--out")          out_path = next();
        else if (a == "--ctx")          n_ctx = atoi(next());
        else if (a == "--prefill")      n_prefill = atoi(next());
        else if (a == "--prefill-rows") n_prefill_rows = atoi(next());
        else if (a == "--ubatch")       n_ubatch = atoi(next());
        else if (a == "--decode")       n_decode = atoi(next());
        else if (a == "--decode-batch") decode_batch = atoi(next());
        else if (a == "--gen")          n_gen = atoi(next());
        else if (a == "--kv-q8")        kv_q8 = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    n_prefill_rows = std::min(n_prefill_rows, n_prefill);

    std::ifstream tf(text_path, std::ios::binary);
    std::stringstream ss; ss << tf.rdbuf();
    const std::string text = ss.str();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(text.size() + 16);
    int n_tok = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
    if (n_tok < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n_tok);
    if (n_tok < n_prefill + n_decode + 1) {
        fprintf(stderr, "text too short: %d tokens, need %d\n", n_tok, n_prefill + n_decode + 1);
        return 1;
    }
    if (n_prefill + n_decode + n_gen + 8 > n_ctx) { fprintf(stderr, "ctx too small\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = std::max(n_ubatch, 2048);
    cp.n_ubatch        = std::max(n_ubatch, 1);
    cp.swa_full        = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.no_perf         = true;
    if (kv_q8) {
        cp.type_k = GGML_TYPE_Q8_0;
        cp.type_v = GGML_TYPE_Q8_0;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }

    FILE * fo = fopen(out_path.c_str(), "wb");
    if (!fo) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
    header h{};
    memcpy(h.magic, "INKL", 4);
    h.n_vocab = n_vocab; h.n_prefill_rows = n_prefill_rows; h.n_decode_rows = n_decode; h.n_gen = n_gen;
    fwrite(&h, sizeof(h), 1, fo);

    llama_batch batch = llama_batch_init(std::max(n_ubatch, 2048), 0, 1);
    std::vector<int32_t> next_ids;
    auto fill = [&](int pos0, int n, int logits_from) {
        batch.n_tokens = n;
        for (int j = 0; j < n; j++) {
            batch.token[j]     = toks[pos0 + j];
            batch.pos[j]       = pos0 + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = (pos0 + j >= logits_from) ? 1 : 0;
        }
    };

    // prefill in ubatches, keep logits of the last n_prefill_rows positions
    double t0 = now_s();
    const int rows_from = n_prefill - n_prefill_rows;
    for (int p = 0; p < n_prefill; p += n_ubatch) {
        const int n = std::min(n_ubatch, n_prefill - p);
        fill(p, n, rows_from);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed (prefill)\n"); return 1; }
        for (int j = 0; j < n; j++) {
            if (p + j >= rows_from) {
                fwrite(llama_get_logits_ith(ctx, j), sizeof(float), n_vocab, fo);
                next_ids.push_back(toks[p + j + 1]);
            }
        }
    }
    llama_synchronize(ctx);
    const double t_prefill = now_s() - t0;

    // teacher-forced decode (single-token or batched T = decode_batch)
    t0 = now_s();
    const float * logits = nullptr;
    for (int i = 0; i < n_decode; i += decode_batch) {
        const int n = std::min(decode_batch, n_decode - i);
        fill(n_prefill + i, n, 0);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        for (int j = 0; j < n; j++) {
            logits = llama_get_logits_ith(ctx, j);
            fwrite(logits, sizeof(float), n_vocab, fo);
            next_ids.push_back(toks[n_prefill + i + j + 1]);
        }
    }
    const double t_decode = now_s() - t0;

    // greedy continuation
    std::vector<int32_t> gen;
    t0 = now_s();
    int pos = n_prefill + n_decode;
    for (int i = 0; i < n_gen && logits; i++) {
        const int best = (int) (std::max_element(logits, logits + n_vocab) - logits);
        gen.push_back(best);
        batch.n_tokens = 1; batch.token[0] = best; batch.pos[0] = pos++; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed (gen)\n"); return 1; }
        logits = llama_get_logits_ith(ctx, 0);
    }
    const double t_gen = now_s() - t0;
    if (!gen.empty()) {
        fwrite(gen.data(), sizeof(int32_t), gen.size(), fo);
    }
    fwrite(next_ids.data(), sizeof(int32_t), next_ids.size(), fo);
    fclose(fo);

    fprintf(stderr, "\n[inkcheck] tokens=%d n_vocab=%d | prefill %d tok in %.3fs (%.1f t/s) | decode %d tok in %.3fs (%.2f t/s) | gen %d tok in %.3fs (%.2f t/s)\n",
            n_tok, n_vocab, n_prefill, t_prefill, n_prefill > 0 ? n_prefill / t_prefill : 0.0,
            n_decode, t_decode, n_decode > 0 ? n_decode / t_decode : 0.0,
            n_gen, t_gen, n_gen > 0 ? n_gen / t_gen : 0.0);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

struct row_stat {
    double lp_a     = 0.0;
    double lp_b     = 0.0;
    double max_abs  = 0.0;
    double kld      = 0.0;
    double p_top_a  = 0.0;
    double dp_top   = 0.0;
    bool   same_top = true;
};

static row_stat cmp_row(const float * a, const float * b, int n, int next_id = -1) {
    row_stat s;
    double ma = -1e30, mb = -1e30;
    int ia = 0, ib = 0;
    for (int i = 0; i < n; i++) {
        s.max_abs = std::max(s.max_abs, (double) fabsf(a[i] - b[i]));
        if (a[i] > ma) { ma = a[i]; ia = i; }
        if (b[i] > mb) { mb = b[i]; ib = i; }
    }
    double za = 0.0, zb = 0.0;
    for (int i = 0; i < n; i++) {
        za += exp((double) a[i] - ma);
        zb += exp((double) b[i] - mb);
    }
    const double lza = log(za) + ma, lzb = log(zb) + mb;
    double kld = 0.0;
    for (int i = 0; i < n; i++) {
        const double la = (double) a[i] - lza, lb = (double) b[i] - lzb;
        kld += exp(la) * (la - lb);
    }
    s.kld      = kld;
    if (next_id >= 0 && next_id < n) {
        s.lp_a = (double) a[next_id] - lza;
        s.lp_b = (double) b[next_id] - lzb;
    }
    s.same_top = (ia == ib);
    s.p_top_a  = exp((double) a[ia] - lza);
    s.dp_top   = fabs(s.p_top_a - exp((double) b[ia] - lzb));
    return s;
}

static void summarize(const char * name, std::vector<row_stat> & st) {
    if (st.empty()) {
        return;
    }
    double max_abs = 0.0, mean_abs = 0.0, max_kld = 0.0, mean_kld = 0.0, max_dp = 0.0, mean_dp = 0.0;
    int same = 0;
    std::vector<double> klds;
    for (auto & s : st) {
        max_abs = std::max(max_abs, s.max_abs); mean_abs += s.max_abs;
        max_kld = std::max(max_kld, s.kld);     mean_kld += s.kld;
        max_dp  = std::max(max_dp, s.dp_top);   mean_dp  += s.dp_top;
        same += s.same_top ? 1 : 0;
        klds.push_back(s.kld);
    }
    std::sort(klds.begin(), klds.end());
    const size_t n = st.size();
    printf("%-8s rows=%zu | top1 igual: %d/%zu (%.2f%%) | max|dlogit|: max=%.3e media=%.3e | KLD: media=%.3e p99=%.3e max=%.3e | d(p_top1): media=%.3e max=%.3e\n",
           name, n, same, n, 100.0 * same / n, max_abs, mean_abs / n, mean_kld / n, klds[std::min(n - 1, (size_t) (0.99 * n))], max_kld, mean_dp / n, max_dp);
    for (size_t i = 0; i < n; i++) {
        if (!st[i].same_top) {
            printf("    fila %zu: top1 distinto (p_top1 ref=%.4f, dp=%.2e, KLD=%.2e, max|dlogit|=%.2e)\n", i, st[i].p_top_a, st[i].dp_top, st[i].kld, st[i].max_abs);
        }
    }
}

static int cmd_cmp(int argc, char ** argv) {
    if (argc < 4) { return 1; }
    FILE * fa = fopen(argv[2], "rb");
    FILE * fb = fopen(argv[3], "rb");
    if (!fa || !fb) { fprintf(stderr, "cannot open inputs\n"); return 1; }
    header ha, hb;
    if (fread(&ha, sizeof(ha), 1, fa) != 1 || fread(&hb, sizeof(hb), 1, fb) != 1) { return 1; }
    const int ha_gen_orig = ha.n_gen;
    if (memcmp(&ha, &hb, sizeof(ha)) != 0) {
        // same row count with another prefill/decode split: compare all rows as one group
        if (ha.n_vocab != hb.n_vocab || ha.n_prefill_rows + ha.n_decode_rows != hb.n_prefill_rows + hb.n_decode_rows) {
            fprintf(stderr, "headers differ: vocab %d/%d prefill %d/%d decode %d/%d gen %d/%d\n",
                    ha.n_vocab, hb.n_vocab, ha.n_prefill_rows, hb.n_prefill_rows, ha.n_decode_rows, hb.n_decode_rows, ha.n_gen, hb.n_gen);
            return 1;
        }
        ha.n_decode_rows += ha.n_prefill_rows; ha.n_prefill_rows = 0; ha.n_gen = 0;
    }
    const int n = ha.n_vocab;
    std::vector<float> a(n), b(n);
    std::vector<row_stat> sp, sd;
    bool bit_exact = true;
    const int n_rows = ha.n_prefill_rows + ha.n_decode_rows;
    // the next-token ids sit after the logits rows and the greedy tokens
    std::vector<int32_t> next_ids(n_rows, -1);
    {
        const long pos = ftell(fa);
        const long off = (long) sizeof(header) + (long) n_rows * n * (long) sizeof(float) + (long) ha_gen_orig * (long) sizeof(int32_t);
        if (fseek(fa, off, SEEK_SET) == 0) {
            if (fread(next_ids.data(), sizeof(int32_t), n_rows, fa) != (size_t) n_rows) {
                std::fill(next_ids.begin(), next_ids.end(), -1);
            }
        }
        fseek(fa, pos, SEEK_SET);
    }
    for (int r = 0; r < n_rows; r++) {
        if (fread(a.data(), sizeof(float), n, fa) != (size_t) n || fread(b.data(), sizeof(float), n, fb) != (size_t) n) { fprintf(stderr, "short read\n"); return 1; }
        bit_exact = bit_exact && memcmp(a.data(), b.data(), sizeof(float) * n) == 0;
        (r < ha.n_prefill_rows ? sp : sd).push_back(cmp_row(a.data(), b.data(), n, next_ids[r]));
    }
    if (next_ids[0] >= 0) {
        // paired difference of the log-likelihood of the real next token
        double sa = 0.0, sb = 0.0, sd1 = 0.0, sd2 = 0.0;
        std::vector<row_stat> all = sp;
        all.insert(all.end(), sd.begin(), sd.end());
        for (auto & s : all) {
            const double d = s.lp_b - s.lp_a;
            sa += s.lp_a; sb += s.lp_b; sd1 += d; sd2 += d * d;
        }
        const double m = (double) all.size();
        const double mean_d = sd1 / m;
        const double se = sqrt(std::max(0.0, sd2 / m - mean_d * mean_d) / m);
        printf("NLL      A=%.5f  B=%.5f  (PPL A=%.4f B=%.4f) | B-A en logp medio: %+.5f +- %.5f (z=%+.2f) sobre %d tokens\n",
               -sa / m, -sb / m, exp(-sa / m), exp(-sb / m), mean_d, se, se > 0 ? mean_d / se : 0.0, (int) m);
    }
    printf("bit-exact: %s\n", bit_exact ? "SI" : "no");
    summarize("prefill", sp);
    summarize("decode", sd);
    if (ha.n_gen > 0) {
        std::vector<int32_t> ga(ha.n_gen), gb(ha.n_gen);
        if (fread(ga.data(), sizeof(int32_t), ha.n_gen, fa) != (size_t) ha.n_gen || fread(gb.data(), sizeof(int32_t), ha.n_gen, fb) != (size_t) ha.n_gen) { return 1; }
        int first = -1;
        for (int i = 0; i < ha.n_gen; i++) {
            if (ga[i] != gb[i]) { first = i; break; }
        }
        if (first < 0) {
            printf("greedy   %d tokens identicos\n", ha.n_gen);
        } else {
            printf("greedy   primera divergencia en el token %d de %d\n", first, ha.n_gen);
        }
    }
    fclose(fa); fclose(fb);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && strcmp(argv[1], "dump") == 0) { return cmd_dump(argc, argv); }
    if (argc >= 2 && strcmp(argv[1], "cmp") == 0)  { return cmd_cmp(argc, argv); }
    fprintf(stderr, "usage: inkcheck dump|cmp ...\n");
    return 1;
}

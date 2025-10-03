#!/usr/bin/env python3
def compute_error_rate(pred_file: str):
    total, errors = 0, 0
    with open(pred_file) as f:
        for line in f:
            line = line.strip()
            # 跳过最后的统计行
            if not line or "files were more probably" in line:
                continue
            pred, fname = line.split()

            # 判断真实类别
            if "gen." in fname:
                gold = "gen.model"
            elif "spam." in fname:
                gold = "spam.model"
            else:
                continue  # 防止有特殊文件名

            total += 1
            if pred != gold:
                errors += 1

    error_rate = errors / total * 100 if total > 0 else 0
    print(f"Total files: {total}")
    print(f"Errors: {errors}")
    print(f"Error rate: {error_rate:.2f}%")

if __name__ == "__main__":
    compute_error_rate("predictions.txt")

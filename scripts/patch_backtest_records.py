with open('src/fpl_oracle/eval/backtest.py', encoding='utf-8') as f:
    content = f.read()

# 1. Add element to all_predictions
old_pos = 'pos_labels = meta_curr["position"].values\n            names = meta_curr["name"].values'
new_pos = 'pos_labels = meta_curr["position"].values\n            names = meta_curr["name"].values\n            elements = meta_curr["element"].values'

content = content.replace(old_pos, new_pos)

old_append = '''                all_predictions.append(
                    {
                        "season": str(s_curr),
                        "round": r_int,
                        "name": str(names[idx]),'''

new_append = '''                all_predictions.append(
                    {
                        "season": str(s_curr),
                        "round": r_int,
                        "element": int(elements[idx]),
                        "name": str(names[idx]),'''

content = content.replace(old_append, new_append)

# 2. Add records_all.csv export and output_dir
old_save = '''        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)'''

new_save = '''        eval_df.to_csv(run_output_dir / "records_all.csv", index=False)
        report_data["output_dir"] = str(run_output_dir)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)'''

content = content.replace(old_save, new_save)

with open('src/fpl_oracle/eval/backtest.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated backtest.py with element and records_all.csv successfully!")

with open("src/fpl_oracle/eval/backtest.py", encoding="utf-8") as f:
    lines = f.readlines()

new_block = [
    "| Position | Count | Model MAE | Base 2 MAE | Gain vs Base | Model RMSE | Base 2 RMSE | Model Spearman | Base 2 Spearman | Coverage (80%) |\n",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n",
    '"""\n',
    "        for p_name, p_met in pos.items():\n",
    '            p_base = p_met.get("baselines", {})\n',
    '            gain = p_met.get("gain_vs_baseline_2_mae", 0.0)\n',
    '            cov = p_met.get("coverage_80_pct", 0.0)\n',
    "            md += (\n",
    "                f\"| **{p_name}** | {p_met.get('count', 0):,} | {p_met.get('model_mae')} | \"\n",
    "                f\"{p_base.get('b2_roll5_mae')} | **{gain:+0.3f}** | {p_met.get('model_rmse')} | \"\n",
    "                f\"{p_base.get('b2_roll5_rmse')} | {p_met.get('model_spearman_tied')} | \"\n",
    "                f\"{p_base.get('b2_roll5_spearman_tied')} | {cov:.1f}% |\\n\"\n",
    "            )\n",
    "\n",
]

# line 622 is index 621, line 634 is index 633
lines[621:634] = new_block

with open("src/fpl_oracle/eval/backtest.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Updated lines 622-634 successfully!")

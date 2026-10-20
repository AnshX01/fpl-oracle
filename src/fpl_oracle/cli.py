"""
FPL Oracle Command Line Interface (CLI).
Interactive terminal commands powered by Typer and Rich.
"""

import asyncio
import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Ensure utf-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

app = typer.Typer(help="FPL Oracle - Local ML-driven 2026/27 Fantasy Premier League Expert")
console = Console(force_terminal=True, legacy_windows=False)


@app.command()
def analyze():
    """Run full team, projection, and transfer analysis."""

    async def _run():
        import pandas as pd

        from fpl_oracle.server.routes.api import get_squad

        result = await get_squad()
        if result.get("status") == "unavailable" or len(result.get("starters", [])) != 11:
            console.print("Configured squad/advice unavailable; no template squad substituted.")
            return
        target_gw = result.get("target_gameweek", result.get("gameweek", result.get("target_gw", "Unknown")))
        lineup = {**result, "starters": pd.DataFrame(result["starters"]), "bench": pd.DataFrame(result["bench"])}

        table = Table(title=f"Optimal Starting Lineup (gameweek {target_gw}) - Formation: {lineup['formation']}")
        table.add_column("Pos", style="cyan")
        table.add_column("Player", style="bold white")
        table.add_column("Cost", justify="right")
        table.add_column("Expected Pts (xP)", justify="right", style="green")
        table.add_column("P10 Floor", justify="right")
        table.add_column("P90 Ceiling", justify="right")
        table.add_column("DefCon Pts", justify="right", style="magenta")

        cap_id = lineup["captain"]["element"]
        vc_id = lineup["vice_captain"]["element"]

        for _, r in lineup["starters"].iterrows():
            badge = " (C)" if r["element"] == cap_id else (" (VC)" if r["element"] == vc_id else "")
            table.add_row(
                r["position"],
                f"{r['web_name']}{badge}",
                f"£{r.get('now_cost', r.get('value', 0) / 10.0):.1f}m",
                f"{r['expected_points']:.2f}",
                f"{r.get('p10', 0.0):.2f}",
                f"{r.get('p90', 0.0):.2f}",
                f"+{r.get('exp_defcon_pts', 0.0):.2f}",
            )

        console.print(table)
        console.print(
            f"[bold]Predicted points with captain: [green]{lineup['total_expected_points']:.1f}[/green][/bold]"
        )

    asyncio.run(_run())


@app.command()
def optimize():
    """Run mathematical MILP transfer optimizer."""

    async def _run():
        from fpl_oracle.server.routes.api import OptimizeRequest, run_optimizer

        opt_res = await run_optimizer(OptimizeRequest())

        console.print(
            Panel(
                f"[bold]Recommendation:[/bold] {opt_res['recommended_plan'].get('recommendation_summary', opt_res['recommended_plan'].get('plan_type', 'Unavailable'))}\n[bold]Hit Verdict:[/bold] {opt_res['hit_verdict']}",
                title="Optimizer Decision",
                border_style="green",
            )
        )

        table = Table(title="Candidate Plans Evaluated")
        table.add_column("Plan Type", style="cyan")
        table.add_column("Summary")
        table.add_column("Net xP", justify="right", style="green")
        table.add_column("Gain vs saving, over the plan", justify="right")
        table.add_column("Hits", justify="right")

        for p in opt_res.get("candidate_plans", []):
            summary = p.get("recommendation_summary", p.get("summary", p.get("plan_type", "Candidate Plan")))
            net_xp = float(p.get("net_expected_points", p.get("net_xp", 0.0)))
            gain = float(p.get("expected_gain", p.get("horizon_gain_vs_roll", 0.0)))
            hits = str(p.get("hits", 0))
            table.add_row(
                p.get("plan_type", "PLAN"),
                summary,
                f"{net_xp:.2f}",
                f"{gain:+.2f}",
                hits,
            )

        console.print(table)

    asyncio.run(_run())


@app.command()
def chips():
    """Display 2026/27 dual-set chip strategy plan."""

    async def _run():
        from fpl_oracle.server.routes.api import get_chip_strategy

        chip_res = await get_chip_strategy()

        if chip_res.get("set_1_deadline_warning"):
            console.print(
                Panel(
                    chip_res["set_1_deadline_warning"],
                    title="[!] 2026/27 Set 1 Chip Expiry Warning",
                    border_style="yellow",
                )
            )

        rec = chip_res.get("recommended_chip")
        console.print(f"[bold]This gameweek:[/bold] {'Play ' + rec if rec else 'Save your chips'}")

        table = Table(title="2026/27 Chip Strategy Schedule")
        table.add_column("Chip", style="bold white")
        table.add_column("Plan week", justify="center", style="green")
        table.add_column("Status", justify="center")

        for c in chip_res.get("chip_plan_table", []):
            gw = c.get("recommended_gw")
            table.add_row(
                str(c.get("chip", c.get("code", "?"))),
                f"Gameweek {gw}" if gw is not None else "No clear benefit",
                str(c.get("status", "")),
            )
        if not chip_res.get("chip_plan_table"):
            console.print("No chip timeline: no chips left to plan, or chip timing is not ready yet.")

        console.print(table)

    asyncio.run(_run())


@app.command()
def briefing():
    """Print the weekly Gameweek Briefing."""

    async def _run():
        from fpl_oracle.briefing.weekly import weekly_briefing_generator

        console.print("[bold green]=== Generating Gameweek Briefing ===[/bold green]")
        briefing_dict = await weekly_briefing_generator.generate_briefing()
        console.print(Markdown(briefing_dict["markdown"]))

    asyncio.run(_run())


@app.command()
def chat():
    """Launch interactive conversational chat with FPL Oracle Expert."""

    async def _run():
        from fpl_oracle.llm.agent import expert_agent

        console.print("[bold green]=== FPL Oracle Expert Chat (Type 'exit' to quit) ===[/bold green]")
        console.print("[dim]Ask about transfers, captaincy, chips, -4 hits, or mini-league tactics.[/dim]\n")

        session_id = "cli_session"
        while True:
            try:
                user_msg = console.input("[bold cyan]You > [/bold cyan]").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["exit", "quit", "q"]:
                    console.print("[dim]Exiting chat session. Good luck this gameweek![/dim]")
                    break

                with console.status("[bold green]Analyzing live FPL data & running projections...[/bold green]"):
                    ans = await expert_agent.answer(user_message=user_msg, session_id=session_id)

                console.print(Panel(Markdown(ans), title="FPL Oracle", border_style="green"))
            except (KeyboardInterrupt, EOFError):
                break

    asyncio.run(_run())


if __name__ == "__main__":
    app()

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

app = typer.Typer(help="FPL Oracle — Local ML-driven 2026/27 Fantasy Premier League Expert")
console = Console(force_terminal=True, legacy_windows=False)


@app.command()
def analyze():
    """Run full team, projection, and transfer analysis."""

    async def _run():
        from fpl_oracle.api.fpl_client import fpl_client
        from fpl_oracle.data.store import data_store
        from fpl_oracle.ml.predict import projection_engine
        from fpl_oracle.optimise.lineup import lineup_optimizer

        console.print("[bold green]=== FPL Oracle 2026/27 Squad Analysis ===[/bold green]")
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        target_gw = next_gw or 6

        console.print(
            f"Current GW: [bold yellow]{curr_gw}[/bold yellow] | Next GW: [bold green]{target_gw}[/bold green]"
        )

        horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures)
        target_df = horizon_proj[target_gw]

        profile = data_store.get_profile()
        user_squad_df = None
        if profile.manager_id:
            try:
                picks, _ = await fpl_client.get_manager_picks(profile.manager_id, curr_gw or 5)
                picks_ids = [p.element for p in picks.picks]
                user_squad_df = target_df[target_df["element"].isin(picks_ids)].copy()
            except Exception:
                pass

        if user_squad_df is None or len(user_squad_df) < 15:
            from fpl_oracle.optimise.squad import squad_optimizer

            user_squad_df = squad_optimizer.solve_best_squad(target_df, budget=1000.0)["squad"].copy()

        lineup = lineup_optimizer.select_lineup_and_captain(user_squad_df)

        table = Table(title=f"Optimal Starting Lineup (GW {target_gw}) — Formation: {lineup['formation']}")
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
                f"£{r['value'] / 10.0:.1f}m",
                f"{r['expected_points']:.2f}",
                f"{r.get('p10', 0.0):.2f}",
                f"{r.get('p90', 0.0):.2f}",
                f"+{r.get('exp_defcon_pts', 0.0):.2f}",
            )

        console.print(table)
        console.print(
            f"[bold]Total Projected Starting Points: [green]{lineup['starters_expected_points']:.2f} xP[/green][/bold]"
        )

    asyncio.run(_run())


@app.command()
def optimize():
    """Run mathematical MILP transfer optimizer."""

    async def _run():
        from fpl_oracle.api.fpl_client import fpl_client
        from fpl_oracle.ml.predict import projection_engine
        from fpl_oracle.optimise.squad import squad_optimizer
        from fpl_oracle.optimise.transfers import transfer_optimizer

        console.print("[bold green]=== FPL Oracle Mathematical Transfer Optimizer ===[/bold green]")
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        target_gw = next_gw or 6

        horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures)
        target_df = horizon_proj[target_gw]

        squad_df = squad_optimizer.solve_best_squad(target_df, budget=1000.0)["squad"].copy()

        opt_res = transfer_optimizer.evaluate_transfer_options(
            current_squad_df=squad_df,
            player_pool_df=target_df,
            bank=5.0,
            free_transfers=1,
            horizon_projections=horizon_proj,
            current_gw=curr_gw or 5,
            target_gw=target_gw,
        )

        console.print(
            Panel(
                f"[bold]Recommendation:[/bold] {opt_res['recommended_plan']['recommendation_summary']}\n[bold]Hit Verdict:[/bold] {opt_res['hit_verdict']}",
                title="Optimizer Decision",
                border_style="green",
            )
        )

        table = Table(title="Candidate Plans Evaluated")
        table.add_column("Plan Type", style="cyan")
        table.add_column("Summary")
        table.add_column("Net xP", justify="right", style="green")
        table.add_column("Net Gain", justify="right")
        table.add_column("Hits", justify="right")

        for p in opt_res["candidate_plans"]:
            table.add_row(
                p["plan_type"],
                p["recommendation_summary"],
                f"{p['net_expected_points']:.2f}",
                f"{p['expected_gain']:+.2f}",
                str(p["hits"]),
            )

        console.print(table)

    asyncio.run(_run())


@app.command()
def chips():
    """Display 2026/27 dual-set chip strategy plan."""

    async def _run():
        from fpl_oracle.api.fpl_client import fpl_client
        from fpl_oracle.chips.planner import chip_planner
        from fpl_oracle.ml.predict import projection_engine
        from fpl_oracle.optimise.squad import squad_optimizer

        console.print("[bold green]=== FPL Oracle 2026/27 Chip Strategy Plan ===[/bold green]")
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()

        horizon_proj = projection_engine.predict_multi_gameweeks(next_gw or 6, 8, boot, fixtures)
        squad_df = squad_optimizer.solve_best_squad(horizon_proj[next_gw or 6], budget=1000.0)["squad"]

        chip_res = chip_planner.generate_chip_strategy(
            current_gw=curr_gw or 5,
            current_squad_df=squad_df,
            horizon_projections=horizon_proj,
            fixtures=fixtures,
            bootstrap=boot,
        )

        if chip_res.get("set_1_deadline_warning"):
            console.print(
                Panel(
                    chip_res["set_1_deadline_warning"],
                    title="[!] 2026/27 Set 1 Chip Expiry Warning",
                    border_style="yellow",
                )
            )

        table = Table(title="2026/27 Chip Strategy Schedule")
        table.add_column("Chip", style="bold white")
        table.add_column("Set", justify="center")
        table.add_column("Rec. GW", justify="center", style="green")
        table.add_column("Exp. Gain", justify="right", style="green")
        table.add_column("Confidence", justify="center")
        table.add_column("Backup GW", justify="center")
        table.add_column("Strategic Role")

        for c in chip_res["chip_plan_table"]:
            table.add_row(
                c["chip"],
                f"Set {c['set']}",
                f"GW {c['recommended_gw']}",
                f"+{c['expected_gain']:.1f} pts",
                c["confidence"],
                f"GW {c['alternative_gw']}",
                c["reasoning"],
            )

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

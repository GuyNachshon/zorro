#!/usr/bin/env python3
"""
Check dataset size and statistics before training.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from icn.data.malicious_extractor import MaliciousExtractor
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def check_malicious_data(dataset_path="malicious-software-packages-dataset"):
    """Check malicious dataset statistics."""
    console.print(Panel.fit("[bold cyan]Checking Malicious Dataset[/bold cyan]", border_style="cyan"))

    try:
        extractor = MaliciousExtractor(dataset_path)
        manifests = extractor.load_manifests()
        categorized = extractor.categorize_packages(manifests)

        table = Table(title="Malicious Packages by Category", show_header=True, header_style="bold red")
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right", style="bold")

        total = 0
        for category, samples in categorized.items():
            table.add_row(category, f"{len(samples):,}")
            total += len(samples)

        table.add_row("[bold]TOTAL", f"[bold]{total:,}", style="bold green")

        console.print(table)
        return total

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 0

def check_benign_data(cache_path="data/benign_samples"):
    """Check benign cache statistics."""
    console.print("\n")
    console.print(Panel.fit("[bold green]Checking Benign Cache[/bold green]", border_style="green"))

    cache_dir = Path(cache_path)
    if not cache_dir.exists():
        console.print(f"[red]Benign cache not found at {cache_path}[/red]")
        return 0

    stats = {}
    total = 0

    for ecosystem in ["npm", "pypi"]:
        ecosystem_dir = cache_dir / ecosystem
        if not ecosystem_dir.exists():
            continue

        for category in ["popular", "longtail"]:
            category_dir = ecosystem_dir / category
            if not category_dir.exists():
                continue

            # Count version directories
            count = sum(1 for pkg_dir in category_dir.iterdir()
                       if pkg_dir.is_dir()
                       for version_dir in pkg_dir.iterdir()
                       if version_dir.is_dir())

            key = f"{ecosystem}/{category}"
            stats[key] = count
            total += count

    table = Table(title="Benign Packages by Source", show_header=True, header_style="bold green")
    table.add_column("Source", style="cyan")
    table.add_column("Count", justify="right", style="bold")

    for source, count in stats.items():
        table.add_row(source, f"{count:,}")

    table.add_row("[bold]TOTAL", f"[bold]{total:,}", style="bold green")

    console.print(table)
    return total

def main():
    """Main function."""
    console.print("\n[bold]📊 Dataset Statistics Check[/bold]\n")

    malicious_count = check_malicious_data()
    benign_count = check_benign_data()

    # Recommendations
    console.print("\n")
    console.print(Panel.fit(
        f"[bold]Dataset Summary[/bold]\n\n"
        f"Malicious Packages: [red]{malicious_count:,}[/red]\n"
        f"Benign Packages: [green]{benign_count:,}[/green]\n"
        f"Total Packages: [cyan]{malicious_count + benign_count:,}[/cyan]\n\n"
        f"[bold]Ratio:[/bold] 1:{benign_count/malicious_count if malicious_count > 0 else 0:.1f} (malicious:benign)",
        border_style="blue",
        title="Summary"
    ))

    # Recommendations
    console.print("\n[bold yellow]💡 Recommendations:[/bold yellow]\n")

    if benign_count < malicious_count * 2:
        console.print(f"[yellow]⚠️  Need more benign data! Recommended: {malicious_count * 3:,} benign packages[/yellow]")
        console.print(f"[yellow]   Run: python collect_more_benign.py --target {malicious_count * 3}[/yellow]\n")

    if malicious_count + benign_count < 5000:
        console.print(f"[yellow]⚠️  Total dataset is small ({malicious_count + benign_count:,} packages)[/yellow]")
        console.print(f"[yellow]   Minimum recommended: 5,000 packages[/yellow]")
        console.print(f"[yellow]   Research quality: 25,000 packages[/yellow]\n")

    if benign_count >= malicious_count * 2 and malicious_count + benign_count >= 5000:
        console.print("[green]✅ Dataset size looks good for training![/green]\n")

        # Suggest training command
        max_mal = min(malicious_count, 5000)
        max_ben = min(benign_count, max_mal * 3)

        console.print(Panel(
            f"[bold]Suggested Training Command:[/bold]\n\n"
            f"python train_neobert.py \\\n"
            f"  --full-pipeline \\\n"
            f"  --max-malicious {max_mal} \\\n"
            f"  --max-benign {max_ben} \\\n"
            f"  --batch-size 24 \\\n"
            f"  --learning-rate 2e-5 \\\n"
            f"  --device cuda \\\n"
            f"  --save-dir checkpoints/neobert_full",
            border_style="green",
            title="🚀 Ready to Train"
        ))

if __name__ == "__main__":
    main()
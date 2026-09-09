"""User administration CLI commands — currently just granting/revoking the
is_admin flag that unlocks the Eval tab's cross-user "All users" view.
There's no other bootstrap/seed mechanism for the very first admin."""

import click
from rich.console import Console

import src.core.db as db
from src.auth.models import User

console = Console()


@click.group()
def admin():
    """Manage user accounts."""
    pass


def _set_admin(email: str, is_admin: bool) -> None:
    db.init_db()
    db._get_engine()
    session = db._SessionLocal()
    try:
        user = session.query(User).filter(User.email == email).first()
        if user is None:
            console.print(f"[red]No user found with email {email}[/red]")
            raise click.Abort()
        user.is_admin = is_admin
        session.commit()
        verb = "granted" if is_admin else "revoked"
        console.print(f"[green]Admin access {verb} for {email}[/green]")
    finally:
        session.close()


@admin.command()
@click.argument("email")
def grant(email: str):
    """Grant admin access (Eval tab's cross-user view) to EMAIL."""
    _set_admin(email, True)


@admin.command()
@click.argument("email")
def revoke(email: str):
    """Revoke admin access from EMAIL."""
    _set_admin(email, False)

import typer

app = typer.Typer()


@app.command()
def greet(name: str) -> str:
    return f"Hello, {name}!"


if __name__ == "__main__":
    app()

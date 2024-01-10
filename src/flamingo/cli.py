import click


@click.group()
def main():
    pass


# need to add the group / command function itself, not the module
main.add_command(simple.driver)


if __name__ == "__main__":
    main()

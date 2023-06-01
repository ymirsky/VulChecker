import json

import click
import networkx as nx


def format_attrs(data):
    formatted_lines = ["<TABLE><TR><TD>Attr</TD><TD>Value</TD></TR><HR/>"]
    for attr, value in data.items():
        formatted_lines.append(f"<TR><TD>{attr}</TD><TD>{value}</TD></TR>")
    formatted_lines.append("</TABLE>")
    return "".join(formatted_lines)


@click.command()
@click.option(
    "--output-file", "-o", type=click.File("w"), default="-", help="Output file"
)
@click.argument("input_file", type=click.File("r"), default="-")
def main(output_file, input_file):
    graph = nx.node_link_graph(json.load(input_file))
    if graph.is_directed():
        print("digraph {", file=output_file)
        edge_sep = "->"
    else:
        print("graph {", file=output_file)
        edge_sep = "--"
    print("overlap = false;", file=output_file)

    for node, data in graph.nodes(data=True):
        label = format_attrs(data)
        print(f'"{node}" [shape=plain, label=<{label}>];', file=output_file)

    for source, target, data in graph.edges(data=True):
        label = format_attrs(data)
        print(f'"{source}" {edge_sep} "{target}" [label=<{label}>];', file=output_file)

    print("}", file=output_file)

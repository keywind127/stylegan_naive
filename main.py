from components.mapping_network import create_mapping_network

def main() -> None:

    mapping_network = create_mapping_network(8, 512)

    mapping_network.summary()

if (__name__ == "__main__"):

    main()
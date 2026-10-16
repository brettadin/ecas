"""Interface between molmass and petitRADTRANS."""
from molmass import ELEMENTS, Formula

from petitRADTRANS.physical_constants import e_molar_mass


def element_number2element_symbol(atomic_number: int) -> str:
    """Convert an element's atomic number (e.g. 6) to its element symbol (e.g. 'C')."""
    for element in ELEMENTS:
        if element.number == atomic_number:
            return element.symbol

    raise ValueError(f"atomic number '{atomic_number}' is not implemented")


def element_symbol2element_number(symbol: str) -> int:
    """Convert an element symbol (e.g. 'C') to its atomic number (e.g. 6)."""
    for element in ELEMENTS:
        if element.symbol == symbol:
            return element.number

    raise ValueError(f"symbol '{symbol}' is not an element symbol")


def get_molmass_name(species: str):
    """Convert a pRT species' name into a molmass-compatible name."""
    from petitRADTRANS.opacities.opacities import Opacity

    species = Opacity.get_species_isotopologue_name(species, join=True)

    name, natural_abundance, charge, _, _, _ = Opacity.split_species_all_info(species)

    # Rearrange isotopes in molmass format
    isotopes = name.split('-')

    if natural_abundance != '':
        return [''.join(isotopes) + charge]

    isotopes.append(charge)

    return isotopes


def get_species_elements(species: str) -> dict[str, int]:
    """Decompose a species into its elements and their amount.

    Example:
        >>> get_species_elements('H2O')
        >>> {'H': 2, 'O': 1}

    Args:
        species: pRT chemical formula of the species

    Returns:
        A dictionary containing the element symbols of the species as keys and their amount as values.
    """
    elements = {}

    names = get_molmass_name(species)

    if len(names) == 1:
        names.append('')

    for name in names[:-1]:
        _elements = Formula(name)._elements

        for _element, info_dict in _elements.items():
            number = tuple(info_dict.values())

            if len(number) > 1:
                raise ValueError(f"molmass element dict for one isotope should have 1 value, but had {len(number)}")

            number = number[0]

            if _element not in elements:
                elements[_element] = number
            else:
                elements[_element] += number

    return elements


def get_species_molar_mass(species: str) -> float:
    """
    Get the molecular mass of a given species.

    This function uses the molmass package to calculate the mass number for the standard isotope of an input species.
    If all_iso is part of the input, it will return the mean molar mass.

    Args:
        species : string
            The chemical formula of the compound. ie C2H2 or H2O
    Returns:
        The molar mass of the compound in atomic mass units.
    """
    if species == 'e-':
        return e_molar_mass

    if "-NatAbund" in species:
        natural_abundance = True
    else:
        natural_abundance = False

    names = get_molmass_name(species)

    molar_mass = 0.0

    if natural_abundance and len(names) == 1:
        return Formula(names[0]).mass
    elif natural_abundance and len(names) != 1:
        raise ValueError(f"species '{species}' has the natural abundance flag, "
                         f"but several isotopes were detected: ({names})")
    elif not natural_abundance:
        charge = None

        # molmass cannot manage formulae containing both '[]'-separated isotopes and charges,
        if names[-1] == '':
            charge = 0.0
        elif names[-1][-1] in ['+', '-']:
            charge = names[-1].replace('_', '')  # get rid of the leading '_'
            charge = charge[-1] + charge[:-1]  # put sign in front of digits

            if len(charge) == 1:
                charge = charge + '1'  # when no charge number is given, assume that it is 1

            charge = float(charge)

        for name in names[:-1]:
            molar_mass += Formula(name).isotope.massnumber

        # Change charge sign ('+' means *less* e-, thanks B. Franklin! https://xkcd.com/567/)
        molar_mass += -charge * e_molar_mass

    return molar_mass

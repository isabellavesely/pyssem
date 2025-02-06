from itertools import combinations
import numpy as np
from sympy import Matrix
from tqdm import tqdm
import math
from utils.collisions.NASA_SBM_frags import frag_col_SBM_vec_lc2

def create_collision_pairs(scen_properties):

    # Get the binomial coefficient of the species
    # This returns all possible combinations of the species
    species =  [species for species_group in scen_properties.species.values() for species in species_group]
    species_cross_pairs = list(combinations(species, 2))
    species_self_pairs = [(s, s) for s in species]

    # Combine the cross and self pairs
    species_pairs = species_cross_pairs + species_self_pairs
    all_collision_pairs = [] 

    # Create unique collision pairs
    collision_pairs_unique = []
    for species_pair in species_pairs:
        collision_pairs_unique.append(SpeciesCollisionPair(species_pair[0], species_pair[1], scen_properties))

    # Then loop through each collision pair and then create a ShellCollisionPair Object
    for collision_pair in collision_pairs_unique:
        species1 = collision_pair.species1
        species2 = collision_pair.species2

        if species1.maneuverable and species2.maneuverable:
            continue

        if species1.mass < scen_properties.LC and species2.mass < scen_properties.LC:
                        continue

        # Loop through the shells
        for i in scen_properties.HMid:
            collision_pair.collision_pair_by_shell.append(ShellCollisionPair(species1, species2, i))

        all_collision_pairs.append(collision_pair)

    print(f"Total amount of unique species pairs is {len(all_collision_pairs)}")

    # Mass Binning

    debris_species = [species for species in scen_properties.species['debris']]

    binC_mass = np.zeros(len(debris_species))
    binE_mass = np.zeros(2 * len(debris_species))
    binW_mass = np.zeros(len(debris_species))
    LBgiven = scen_properties.LC

    for index, debris in enumerate(debris_species):
        binC_mass[index] = debris.mass
        binE_mass[2 * index: 2 * index + 2] = [debris.mass_lb, debris.mass_ub]
        binW_mass[index] = debris.mass_ub - debris.mass_lb

    binE_mass = np.unique(binE_mass)

    for i, species_pair in tqdm(enumerate(all_collision_pairs), total=len(all_collision_pairs), desc="Processing species pairs"):
         for shell_collision in species_pair.collision_pair_by_shell:
            gamma = calculate_gamma_for_collision_pair(i, shell_collision, scen_properties, debris_species, binE_mass, LBgiven)


def calculate_gamma_for_collision_pair(i, collision_pair, scen_properties, debris_species, binE_mass, LBgiven):
    """
        Using evolve_bin, calculate the number of fragments for each collision. 
    """
    m1, m2 = collision_pair.species1.mass, collision_pair.species2.mass
    r1, r2 = collision_pair.species1.radius, collision_pair.species2.radius
    v = scen_properties.v_imp

    if m1 < scen_properties.LC or m2 < scen_properties.LC:
        collision_pair.fragments = None
        return collision_pair

    fragments = evolve_bins_nasa_sbm(scen_properties, m1, m2, r1, r2, v, binE_mass, collision_index=i, n_shells=0)

    collision_pair.fragments = fragments

    return collision_pair

def evolve_bins_nasa_sbm(scen_properties, m1, m2, r1, r2, v, binE_mass, collision_index, n_shells=0):
            
    # Need to now follow the NASA SBM route, first we need to create p1_in and p2_in
    #  Parameters:
    # - ep: Epoch
    # - p1_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
    # - p2_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
    # - param: Dictionary containing parameters like 'max_frag', 'mu', 'req', 'maxID', etc.
    # - LB: Lower bound for fragment sizes (meters)
    # Super sampling ratio
    SS = 20

    p1_in = np.array([
        1250.0,  # mass in kg
        4.0,     # radius in meters
        2372.4,  # r_x in km, 1000 km
        2743.1,  # r_y in km
        6224.8,  # r_z in km
        -5.5,    # v_x in km/s
        -3.0,    # v_y in km/s
        3.8,     # v_z in km/s
        1      # object_class (dimensionless)
    ])

    p2_in = np.array([
        6.0,     # mass in kg
        0.1,     # radius in meters
        2372.4,  # r_x in km
        2743.1,  # r_y in km
        6224.8,  # r_z in km
        3.2,     # v_x in km/s
        5.4,     # v_y in km/s
        -3.9,    # v_z in km/s
        1      # object_class (dimensionless)
    ])

    param = {
        'req': 6.3781e+03,
        'mu': 3.9860e+05,
        'j2': 0.0011,
        'max_frag': float('inf'),  # Inf in MATLAB translates to float('inf') in Python
        'maxID': 0,
        'density_profile': 'static'
    }
    
    altitude = scen_properties.HMid[collision_index] 
    earth_radius = 6371  # Earth's mean radius in km
    latitude_deg = 45  # in degrees
    longitude_deg = 60  # in degrees

    # Convert degrees to radians
    latitude_rad = math.radians(latitude_deg)
    longitude_rad = math.radians(longitude_deg)

    # Compute the radial distance from Earth's center
    r = earth_radius + altitude

    # Calculate the position vector in ECEF coordinates
    x = r * math.cos(latitude_rad) * math.cos(longitude_rad)
    y = r * math.cos(latitude_rad) * math.sin(longitude_rad)
    z = r * math.sin(latitude_rad)

    # Return the position vector
    x, y, z

    # up to correct mass too
    if m1 < m2:
        m1, m2 = m2, m1
        r1, r2 = r2, r1

    p1_in[0], p2_in[0] = m1, m2 
    p1_in[1], p2_in[1] = r1, r2

    # remove a from r_x from both p1_in and p2_in
    # the initial norm is 1000, so we need to remove the difference
    p1_in[2], p1_in[3], p1_in[4] = x, y, z 
    p2_in[2], p2_in[3], p2_in[4] = x, y, z
        
    LB = 0.1

    debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc2(0, p1_in, p2_in, param, LB)

    print(len(debris1), len(debris2))

    # Loop through 
    frag_a = []
    frag_mass = []

    for debris in debris1:
        norm_earth_radius = debris[0]
        if norm_earth_radius < 1:
            continue  # decayed

        frag_a.append((norm_earth_radius - 1) * 6371)
        frag_mass.append(debris[7])
    
    for debris in debris2:
        norm_earth_radius = debris[0]
        if norm_earth_radius < 1:
            continue  # decayed

        frag_a.append((norm_earth_radius - 1) * 6371)
        frag_mass.append(debris[7])

    frag_properties = np.array([frag_a, frag_mass]).T

    binE_alt = np.linspace(scen_properties.min_altitude, scen_properties.max_altitude, n_shells + 1)  # Bin edges

    bins = [binE_alt, binE_mass]

    hist, edges = np.histogramdd(frag_properties, bins=bins)

    hist = hist / (SS * 2)

    return hist



class ShellCollisionPair:
    def __init__(self, species1, species2, index) -> None:
        self.species1 = species1
        self.species2 = species2
        self.shell_index = index
        self.s1_col_sym_name = f"{species1.sym_name}_sh_{index}"
        self.s2_col_sym_name = f"{species2.sym_name}_sh_{index}"
        self.col_sym_name = f"{species1.sym_name}__{species2.sym_name}_sh_{index}" 

        self.gamma = None 
        self.fragments = None
        self.catastrophic = None
        self.binOut = None
        self.altA = None
        self.altE = None

        self.debris_eccentrcity_bins = None


class SpeciesCollisionPair:
    def __init__(self, species1, species2, scen_properties) -> None:
        self.species1 = species1
        self.species2 = species2
        self.collision_pair_by_shell = []
        self.collision_processed = []

        # # Create a matrix of gammas, rows are the shells, columns are debris species (only 2 as in loop)
        self.gamma = Matrix(scen_properties.n_shells, 2, lambda i, j: -1)

        # Create a list of source sinks, first two are the active species then the active speices
        self.source_sinks = [species1, species2] + scen_properties.species['debris']

        # Implementing logic for gammas calculations based on species properties
        if species1.maneuverable and species2.maneuverable:
            # Multiplying each element in the first column of gammas by the product of alpha_active values
            self.gamma[:, 0] = self.gamma[:, 0] * species1.alpha_active * species2.alpha_active
            if species1.slotted and species2.slotted:
                # Applying the minimum slotting effectiveness if both are slotted
                self.gamma[:, 0] = self.gamma[:, 0] * min(species1.slotting_effectiveness, species2.slotting_effectiveness)

        elif (species1.maneuverable and not species2.maneuverable) or (species2.maneuverable and not species1.maneuverable):
            if species1.trackable and species2.maneuverable:
                self.gamma[:, 0] = self.gamma[:, 0] * species2.alpha
            elif species2.trackable and species1.maneuverable:
                self.gamma[:, 0] = self.gamma[:, 0] * species1.alpha

        # Applying symmetric loss to both colliding species
        self.gamma[:, 1] = self.gamma[:, 0]

        # Rocket Body Flag - 1: RB; 0: not RB
        # Will be 0 if both are None type
        if species1.RBflag is None and species2.RBflag is None:
            self.RBflag = 0
        elif species2.RBflag is None:
            self.RBflag = species2.RBflag
        elif species2.RBflag is None:
            self.RBflag = species1.RBflag
        else:
            self.RBflag = max(species1.RBflag, species2.RBflag)

        self.phi = None # Proabbility of collision, based on v_imp, shell volumen and object_radii
        self.catastrophic = None # Catastrophic flag for each shell
        self.eqs = None # All Symbolic equations 
        self.nf = None # Initial symbolic equations for the number of fragments
        self.sigma = None # Square of the impact parameter



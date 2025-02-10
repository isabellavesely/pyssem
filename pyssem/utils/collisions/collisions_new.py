from itertools import combinations
import numpy as np
from sympy import Matrix, pi, S, zeros
from tqdm import tqdm
import math
import re
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
        for i in range(len(scen_properties.HMid)):
            collision_pair.collision_pair_by_shell.append(ShellCollisionPair(species1, species2, i)) # i is your collision shell index

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
            gamma = calculate_gamma_for_collision_pair(shell_collision.shell_index, shell_collision, scen_properties, debris_species, binE_mass, LBgiven)
            species_pair.collision_processed.append(gamma)

    # find the unique mass - this should be done in the scenario properties >>>> This is required for elliptical orbits
    debris_species_names = [species.sym_name for species in scen_properties.species['debris']]
    # pattern = re.compile(r'^N_[^_]+kg')
    # unique_debris_names = set()

    # for name in debris_species_names:
    #     match = pattern.match(name)
    #     if match:
    #         unique_debris_names.add(match.group())
    # unique_debris_names = list(unique_debris_names)

    return generate_collision_equations(all_collision_pairs, scen_properties, mass_bins=debris_species_names)


def generate_collision_equations(all_collision_pairs, scen_properties, mass_bins):

    # Ensure that debris species variables are included in 'all_symbolic_vars'
    all_symbolic_vars = scen_properties.all_symbolic_vars

    # Rebuild the mappings to include the debris species
    species_name_to_idx = {str(var): idx for idx, var in enumerate(all_symbolic_vars)}
    species_sym_vars = {str(var): var for var in all_symbolic_vars}

    n_species = len(all_symbolic_vars)
    equations = Matrix.zeros(scen_properties.n_shells, n_species)

    phi_matrix = None

    for collision_pair in tqdm(all_collision_pairs, total=len(all_collision_pairs), desc="Generating collision equations"):

        # First set the eqs to make sure not none
        collision_pair.eqs = Matrix(scen_properties.n_shells, scen_properties.species_length, lambda i, j: 0)

        # Check if there is no collisions between species
        if len(collision_pair.collision_pair_by_shell) == 0:
            continue

        # Map out the major components
        species1 = collision_pair.species1
        species2 = collision_pair.species2
        species1_name = species1.sym_name
        species2_name = species2.sym_name

        # Conversion factor
        meter_to_km = 1 / 1000

        # Compute collision cross-section (sigma)
        collision_pair.sigma = (species1.radius * meter_to_km + species2.radius * meter_to_km) ** 2

        # Compute collision rate (phi) for each shell
        collision_pair.phi = (
            pi * scen_properties.v_imp2
            / (scen_properties.V * meter_to_km**3)
            * collision_pair.sigma
            * S(86400)
            * S(365.25)
        )
        phi_matrix = Matrix(collision_pair.phi)

        # If phi_matrix is a scalar, convert it into a 1x1 matrix
        if phi_matrix.shape == ():  # Single scalar case
            phi_matrix = Matrix([phi_matrix])  # Convert scalar to 1x1 matrix
        else:
            # If phi_matrix is a 1D row or flat matrix, reshape it into a column vector (n, 1)
            phi_matrix = phi_matrix.reshape(len(phi_matrix), 1) 

        # Now loop through the shells, get the gammas.  
        for shell_collision in collision_pair.collision_pair_by_shell:
            # Get the current state of the gamma (to add to)
            shell_collision.gamma = collision_pair.gamma.copy()

            s_source = shell_collision.shell_index  # Source shell index (0-based)

            # Get species variable names including shell number
            species1_var_name = f'{species1_name}_{s_source + 1}'
            species2_var_name = f'{species2_name}_{s_source + 1}'

            # Get symbolic variables
            N_species1_s = species_sym_vars.get(species1_var_name)
            N_species2_s = species_sym_vars.get(species2_var_name)

            if N_species1_s is None or N_species2_s is None:
                continue  # Skip if species variables are not found

            phi_s = collision_pair.phi[s_source]

            fragments = shell_collision.fragments

            if fragments is None:
                continue

            n_destination_shells, n_mass_bins = fragments.shape

            for s_destination in range(n_destination_shells):
                fragments_sd = fragments[s_destination]  # Fragments ending up in shell s_destination
                for mass_bin_index in range(n_mass_bins):
                    num_frags = fragments_sd[mass_bin_index]  # Number of fragments in this mass bin
                    if num_frags != 0:
                        mass_value = mass_bins[mass_bin_index]
                        # Generate debris species variable name including destination shell number
                        debris_species_name = f'{mass_value}_{s_destination + 1}'
                        idx_debris = species_name_to_idx.get(debris_species_name)
                        if idx_debris is not None:
                            debris_var = species_sym_vars[debris_species_name]
                            # Compute delta gain for debris species (symbolic)
                            delta_gain = phi_s * N_species1_s * N_species2_s * num_frags
                            equations[s_destination, idx_debris] += delta_gain  

            # Now, extract the equations for the debris species and store them in a (10, 20) matrix
            # Initialize the debris equations matrix
            debris_length = len(mass_bins)  # No longer multiplying by len(ecc_bins)
            equations_debris = Matrix.zeros(scen_properties.n_shells, debris_length)

            # # Map debris species variable names to indices within the shell
            debris_species_idx_within_shell = {}
            for idx_within_shell, debris_species_name in enumerate(mass_bins[:debris_length]):
                # base_name = debris_species_name.split('_')[0]  # Only use mass_value, no eccentricity
                debris_species_idx_within_shell[debris_species_name] = idx_within_shell

            # Populate the debris equations matrix
            for s in range(scen_properties.n_shells):
                for mass_value in mass_bins:
                    debris_species_name = f'{mass_value}_{s + 1}'  # No eccentricity component
                    base_name = f'{mass_value}'  # Only mass value
                    idx_species = species_name_to_idx.get(debris_species_name)
                    idx_debris = debris_species_idx_within_shell.get(base_name)
                    if idx_species is not None and idx_debris is not None:
                        eq = equations[s, idx_species]
                        equations_debris[s, idx_debris] += eq

            species_names = scen_properties.species_names
            species1_idx = species_names.index(species1.sym_name)
            species2_idx = species_names.index(species2.sym_name)

            # Find the start of the debris species index, which will be the first item in species_names that starts with 'N_'
            debris_start_idx = next(i for i, name in enumerate(species_names) if name.startswith('N_'))

            try:
                eq_s1 = collision_pair.gamma[:, 1].multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)
                eq_s2 = collision_pair.gamma[:, 1].multiply_elementwise(phi_matrix).multiply_elementwise(species1.sym).multiply_elementwise(species2.sym)
            except Exception as e:
                print(f"Exception caught: {e}")
                print("Error in multiplying gammas, check that each component is a column vector and correct shape.")

            # eqs = Matrix(zeros(scen_properties.n_shells, len(scen_properties.species['debris'])+2))
            eqs = Matrix(zeros(scen_properties.n_shells, scen_properties.species_length))

            # Add in eq_1 at species1_idx and eq_2 at species2_idx
            eqs[:, species1_idx] = eq_s1
            eqs[:, species2_idx] += eq_s2

            # Loop through each debris species
            for i in range(len(scen_properties.species['debris'])):
                # Calculate the corresponding index in the overall species list
                deb_index = debris_start_idx + i
                # Assign the columns from equations_debris to the appropriate columns in eqs
                eqs[:, deb_index] = equations_debris[:, i]

            collision_pair.eqs = eqs

    return all_collision_pairs



def calculate_gamma_for_collision_pair(collision_index, collision_pair, scen_properties, debris_species, binE_mass, LBgiven):
    """
        Using evolve_bin, calculate the number of fragments for each collision. 
    """
    m1, m2 = collision_pair.species1.mass, collision_pair.species2.mass
    r1, r2 = collision_pair.species1.radius, collision_pair.species2.radius
    v = scen_properties.v_imp

    if m1 < scen_properties.LC or m2 < scen_properties.LC:
        collision_pair.fragments = None
        return collision_pair

    fragments = evolve_bins_nasa_sbm(scen_properties, m1, m2, r1, r2, v, binE_mass, collision_index=collision_index)

    collision_pair.fragments = fragments

    return collision_pair

def adjust_altitude(p, altitude):
    r_vec = p[2:5]  # Extract position vector (r_x, r_y, r_z)
    r_current = np.linalg.norm(r_vec)  # Compute current radius
    r_new = 6378 + altitude  # New radius based on altitude
    scale_factor = r_new / r_current  # Scaling factor
    
    # Scale the position vector
    p_new = p.copy()  # Copy the original object
    p_new[2:5] = scale_factor * r_vec  # Adjust position
    
    return p_new

def evolve_bins_nasa_sbm(scen_properties, m1, m2, r1, r2, v, binE_mass, collision_index):
            
    # Need to now follow the NASA SBM route, first we need to create p1_in and p2_in
    #  Parameters:
    # - ep: Epoch
    # - p1_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
    # - p2_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
    # - param: Dictionary containing parameters like 'max_frag', 'mu', 'req', 'maxID', etc.
    # - LB: Lower bound for fragment sizes (meters)
    # Super sampling ratio
    SS = 20

    p1 = np.array([
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

    p2 = np.array([
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
    
    # The stardard code is for 1000km alt. Needs to be adjusted for the shell. 
    altitude = scen_properties.HMid[collision_index] 
    p1_in = adjust_altitude(p1, altitude)
    p2_in = adjust_altitude(p2, altitude)  

    # Correct mass and raidus
    if m1 < m2:
        m1, m2 = m2, m1
        r1, r2 = r2, r1

    p1_in[0], p2_in[0] = m1, m2 
    p1_in[1], p2_in[1] = r1, r2
        
    LB = 0.1

    # Attempt to call frag_col_SBM_vec_lc2 up to 5 times before failing
    # This is because for smaller objects hitting (above LC) it may fail to produce any fragments. 
    attempts = 5
    for attempt in range(attempts):
        try:
            debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc2(0, p1_in, p2_in, param, LB, scen_properties.min_altitude + 6378)
            break  # If successful, break out of the loop
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == attempts - 1:
                raise RuntimeError("Failed to call frag_col_SBM_vec_lc2 after 5 attempts")


    # print(len(debris1), len(debris2))

    # Loop through 
    frag_altitude = []
    frag_mass = []

    for debris in debris1:
        norm_earth_radius = debris[0]
        if norm_earth_radius < 1:
            continue  # decayed

        frag_altitude.append(debris[0] - 6378)
        frag_mass.append(debris[7])
    
    for debris in debris2:
        norm_earth_radius = debris[0]
        if norm_earth_radius < 1:
            continue  # decayed

        frag_altitude.append(debris[0] - 6378)
        frag_mass.append(debris[7])

    frag_properties = np.array([frag_altitude, frag_mass]).T

    #binE_alt = np.linspace(scen_properties.min_altitude, scen_properties.max_altitude, n_shells + 1)  # Bin edges
    binE_alt = scen_properties.R0_km
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



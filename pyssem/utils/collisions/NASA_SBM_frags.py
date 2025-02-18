import numpy as np
from utils.collisions.collisions import func_Am, func_dv
from poliastro.core.elements import rv2coe
from astropy import units as u
from poliastro.bodies import Earth
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import pickle


def frag_col_SBM_vec_lc2(ep, p1_in, p2_in, param, LB, minimum_altitude = 0, orbit_type='LEO'):
    """
    Collision model following NASA EVOLVE 4.0 standard breakup model (2001)
    with the revision in ODQN "Proper Implementation of the 1998 NASA Breakup Model" (2011)

    The Debris output looks lke this:
    1. `a`: Semi-major axis (in km) - defines the size of the orbit.
    2. `ecco`: Eccentricity - describes the shape of the orbit (0 for circular, closer to 1 for highly elliptical).
    3. `inclo`: Inclination (in degrees) - the tilt of the orbit relative to the equatorial plane.
    4. `nodeo`: Right ascension of ascending node (in degrees) - describes the orientation of the orbit in the plane.
    5. `argpo`: Argument of perigee (in degrees) - the angle between the ascending node and the orbit's closest point to Earth.
    6. `mo`: Mean anomaly (in degrees) - the position of the debris along the orbit at a specific time.
    7. `bstar`: Drag coefficient - relates to atmospheric drag affecting the debris.
    8. `mass`: Mass of the debris fragment (in kg).
    9. `radius`: Radius of the debris fragment (in meters).
    10. `errors`: Error values (various metrics used for error tracking or uncertainty).
    11. `controlled`: A flag indicating whether the fragment is controlled (e.g., through active debris removal or a controlled reentry).
    12. `a_desired`: Desired semi-major axis (in km) - often used in mission planning.
    13. `missionlife`: Expected mission lifetime (in years).
    14. `constel`: Identifier for the constellation to which the debris belongs (if any).
    15. `date_created`: Date the fragment data was created (format depends on implementation, e.g., `YYYY-MM-DD`).
    16. `launch_date`: Date of launch of the parent object (format depends on implementation).
    17. `r[idx_a]`: Position vector (in km) - describes the 3D position of the fragment in space.
    18. `v[idx_a]`: Velocity vector (in km/s) - describes the 3D velocity of the fragment.
    19. `frag_objectclass`: Object class - a dimensionless identifier describing the type of fragment or debris.
    20. `ID_frag`: Unique identifier for the fragment (could be a string or numeric ID).
    
    Parameters:
    - ep: Epoch
    - p1_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
    - p2_in: Array containing [mass, radius, r_x, r_y, r_z, v_x, v_y, v_z, object_class]
    - param: Dictionary containing parameters like 'max_frag', 'mu', 'req', 'maxID', etc.
    - LB: Lower bound for fragment sizes (meters)
    
    Returns:
    - debris1: Array containing debris fragments from parent 1
    - debris2: Array containing debris fragments from parent 2
    - isCatastrophic: Boolean indicating if the collision is catastrophic
    """
    
    # Ensure p1_mass > p2_mass, or p1_radius > p2_radius if p1_mass == p2_mass
    if p1_in[0] < p2_in[0] or (p1_in[0] == p2_in[0] and p1_in[1] < p2_in[1]):
        p1_in, p2_in = p2_in, p1_in
    
    # Extract parameters for p1 and p2
    p1_mass, p1_radius = p1_in[0], p1_in[1]
    p1_r, p1_v = p1_in[2:5], p1_in[5:8]
    p1_objclass = p1_in[8]
    
    p2_mass, p2_radius = p2_in[0], p2_in[1]
    p2_r, p2_v = p2_in[2:5], p2_in[5:8]
    p2_objclass = p2_in[8]
    
    # Compute relative velocity (dv) and catastrophic ratio
    dv = np.linalg.norm(np.array(p1_v) - np.array(p2_v))  # km/s
    catastroph_ratio = (p2_mass * (dv * 1000) ** 2) / (2 * p1_mass * 1000)  # J/g
    
    # Determine if collision is catastrophic
    if catastroph_ratio < 40:
        M = p2_mass * dv ** 2
        isCatastrophic = False
    else:
        M = p1_mass + p2_mass
        isCatastrophic = True
    
    # Create debris size distribution
    dd_edges = np.logspace(np.log10(LB), np.log10(min(1, 2 * p1_radius)), 200)
    log10_dd = np.log10(dd_edges)
    dd_means = 10 ** (log10_dd[:-1] + np.diff(log10_dd) / 2)
    
    nddcdf = 0.1 * M ** 0.75 * dd_edges ** (-1.71)
    ndd = np.maximum(0, -np.diff(nddcdf))
    floor_ndd = np.floor(ndd).astype(int)
    rand_sampling = np.random.rand(len(ndd))
    add_sampling = rand_sampling > (1 - (ndd - floor_ndd))
    # d_pdf = np.repeat(dd_means, floor_ndd + add_sampling.astype(int))
    d_pdf = np.repeat(dd_means, np.floor(floor_ndd + add_sampling).astype(int))

    
    # Shuffle the diameters
    d = np.random.permutation(d_pdf)

    # plt.figure(50)
    # plt.clf()

    # # Plot histogram
    # plt.hist(d, bins=dd_edges, log=True, edgecolor='black', alpha=0.7)

    # # Plot line
    # plt.plot(dd_means, ndd[:-1], 'k')

    # # Set log scale for both axes
    # plt.xscale('log')
    # plt.yscale('log')

    # # Set labels and title
    # plt.xlabel('d [m]')
    # plt.ylabel('Number of fragments [-]')
    # plt.title('Number of fragments vs diameter')

    # # Show the plot
    # plt.savefig('Number_of_frags_vs_diameter.png')
    
    # Calculate mass of fragments
    A = 0.556945 * d ** 2.0047077
    Am = func_Am(d, p1_objclass)
    m = A / Am
    
    # print(f"Number of fragments: {len(m)}")
    
    # Initialize remnant indices
    idx_rem1 = np.array([], dtype=int)
    idx_rem2 = np.array([], dtype=int)
    
    if np.sum(m) < M:
        if isCatastrophic:
            # -----------------------------
            # Catastrophic collision handling
            # -----------------------------
            # In MATLAB:
            # largeidx = (d > p2_radius*2 | m > p2_mass) & d < p1_radius*2;
            largeidx = ((d > 2 * p2_radius) | (m > p2_mass)) & (d < 2 * p1_radius)
            m_assigned_large = max(0, np.sum(m[largeidx]))
        
            if m_assigned_large > p1_mass:
                idx_large = np.where(largeidx)[0]
                # Sort by mass (ascending)
                dord1 = np.argsort(m[idx_large])
                cumsum_m1 = np.cumsum(m[idx_large][dord1])
                indices = np.where(cumsum_m1 < p1_mass)[0]
                if indices.size > 0:
                    lastidx1 = indices[-1]
                    # Remove fragments that would push mass over p1_mass
                    to_remove = idx_large[dord1[lastidx1+1:]]
                    m = np.delete(m, to_remove)
                    d = np.delete(d, to_remove)
                    A = np.delete(A, to_remove)
                    Am = np.delete(Am, to_remove)
                    largeidx = np.delete(largeidx, to_remove)
                    m_assigned_large = cumsum_m1[lastidx1]
                else:
                    m_assigned_large = 0
        
            mass_max_small = min(p2_mass, m_assigned_large)
        
            # For fragments not already assigned to the large satellite
            smallidx_temp = np.where(~largeidx)[0]
            dord = np.argsort(m[smallidx_temp])
            cumsum_m = np.cumsum(m[smallidx_temp][dord])
            indices = np.where(cumsum_m <= mass_max_small)[0]
            if indices.size > 0:
                lastidx_small = indices[-1]
                small_indices = smallidx_temp[dord[:lastidx_small+1]]
                smallidx = np.zeros(len(d), dtype=bool)
                smallidx[small_indices] = True
                m_assigned_small = max(0, np.sum(m[smallidx]))
            else:
                m_assigned_small = 0
                smallidx = np.zeros(len(d), dtype=bool)
        
            m_remaining_large = p1_mass - m_assigned_large
            m_remaining_small = p2_mass - m_assigned_small
            m_remaining = [m_remaining_large, m_remaining_small]
        
            # --- Remnant mass distribution ---
            m_remSum = M - np.sum(m)
            # Randomly pick a number between 2 and 8 (inclusive)
            num_rand = np.random.randint(2, 9)
            remDist = np.random.rand(num_rand)
            m_rem_temp = m_remSum * remDist / np.sum(remDist)
            num_rem = len(m_rem_temp)
        
            # Sort remnant masses in descending order
            idx_sort = np.argsort(m_rem_temp)[::-1]
            m_rem_sort = m_rem_temp[idx_sort].copy()
            # Randomly assign each remnant to satellite 1 or 2:
            # In MATLAB: 1+round(rand(num_rem,1)) yields 1 or 2.
            rem_temp_ordered = 1 + np.round(np.random.rand(num_rem)).astype(int)
        
            for i_rem in range(num_rem):
                # assign: 1 means larger (index 0), 2 means smaller (index 1)
                assign_idx = 0 if (rem_temp_ordered[i_rem] == 1) else 1
                if m_rem_sort[i_rem] > m_remaining[assign_idx]:
                    diff_mass = m_rem_sort[i_rem] - m_remaining[assign_idx]
                    m_rem_sort[i_rem] = m_remaining[assign_idx]
                    m_remaining[assign_idx] = 0
                    if i_rem == num_rem - 1:
                        m_rem_sort = np.append(m_rem_sort, diff_mass)
                        rem_temp_ordered = np.append(rem_temp_ordered, 2 if assign_idx == 0 else 1)
                        num_rem += 1
                    else:
                        m_rem_sort[i_rem+1:] += diff_mass / (num_rem - i_rem - 1)
                        # Reassign all subsequent remnants to the other satellite
                        rem_temp_ordered[i_rem+1:] = (2 if assign_idx == 0 else 1)
                else:
                    m_remaining[assign_idx] -= m_rem_sort[i_rem]
        
            m_rem = m_rem_sort
            # Compute approximate diameters for remnants using the parent’s density
            d_rem_approx = np.zeros_like(m_rem)
            rem1_temp = (rem_temp_ordered == 1)
            d_rem_approx[rem1_temp] = ((m_rem[rem1_temp] / p1_mass * p1_radius**3) ** (1/3)) * 2
            d_rem_approx[~rem1_temp] = ((m_rem[~rem1_temp] / p2_mass * p2_radius**3) ** (1/3)) * 2
            Am_rem = func_Am(d_rem_approx, p1_objclass)
            A_rem = m_rem * Am_rem
            d_rem = d_rem_approx
            idx_rem1 = np.where(rem1_temp)[0]
            idx_rem2 = np.where(~rem1_temp)[0]
        
        else:
            # -----------------------------
            # Non-catastrophic collision handling
            # -----------------------------
            largeidx = ((d > 2 * p2_radius) | (m > p2_mass)) & (d < 2 * p1_radius)
            m_assigned_large = max(0, np.sum(m[largeidx]))
        
            if m_assigned_large > p1_mass:
                idx_large = np.where(largeidx)[0]
                dord1 = np.argsort(m[idx_large])
                cumsum_m1 = np.cumsum(m[idx_large][dord1])
                indices = np.where(cumsum_m1 < p1_mass)[0]
                if indices.size > 0:
                    lastidx1 = indices[-1]
                    to_remove = idx_large[dord1[lastidx1+1:]]
                    m = np.delete(m, to_remove)
                    d = np.delete(d, to_remove)
                    A = np.delete(A, to_remove)
                    Am = np.delete(Am, to_remove)
                    largeidx = np.delete(largeidx, to_remove)
                    m_assigned_large = cumsum_m1[lastidx1]
                else:
                    # If no valid index found, set mass to the cumulative mass or zero.
                    m_assigned_large = 0
        
            m_remaining_large = p1_mass - m_assigned_large
            smallidx = (d > 2 * p1_radius) & (~largeidx)
        
            m_remSum = M - np.sum(m)
            if m_remSum > m_remaining_large:
                m_rem1 = m_remaining_large
                d_rem_approx1 = ((m_rem1 / p1_mass * p1_radius**3) ** (1/3)) * 2
                m_rem2 = m_remSum - m_remaining_large
                d_rem_approx2 = ((m_rem2 / p2_mass * p2_radius**3) ** (1/3)) * 2
                d_rem_approx = np.concatenate((np.array([d_rem_approx1]), np.array([d_rem_approx2])))
                m_rem = np.concatenate((np.array([m_rem1]), np.array([m_rem2])))
                Am_rem = func_Am(d_rem_approx, p1_objclass)
                A_rem = m_rem * Am_rem
                d_rem = d_rem_approx
                idx_rem1 = np.array([0])
                idx_rem2 = np.array([1])
            else:
                m_rem = np.array([m_remSum])
                d_rem_approx = ((m_rem / p1_mass * p1_radius**3) ** (1/3)) * 2
                Am_rem = func_Am(d_rem_approx, p1_objclass)
                A_rem = m_rem * Am_rem
                d_rem = d_rem_approx
                idx_rem1 = np.array([0])
                idx_rem2 = np.array([])
        
            # Remove remnants that are too small (less than LB or less than 0.1% of M)
            idx_too_small = np.where((d_rem < LB) & (m_rem < M/1000))[0]
            m_rem = np.delete(m_rem, idx_too_small)
            d_rem = np.delete(d_rem, idx_too_small)
            A_rem = np.delete(A_rem, idx_too_small)
            Am_rem = np.delete(Am_rem, idx_too_small)
            # Adjust indices if a remnant was removed
            if idx_too_small.size > 0:
                if np.array_equal(idx_too_small, np.array([0])):
                    idx_rem1 = np.array([])
                elif np.array_equal(idx_too_small, np.array([1])):
                    idx_rem2 = np.array([])
        
    else:
        # -----------------------------
        # sum(m) >= M case
        # -----------------------------
        sort_idx_mass = np.argsort(m)
        cumsum_m = np.cumsum(m[sort_idx_mass])
        indices = np.where(cumsum_m < M)[0]
        if indices.size > 0:
            lastidx = indices[-1]
            valididx = sort_idx_mass[:lastidx+1]
            m = m[valididx]
            d = d[valididx]
            A = A[valididx]
            Am = Am[valididx]
        else:
            m = np.array([])
            d = np.array([])
            A = np.array([])
            Am = np.array([])
        
        largeidx = ((d > 2 * p2_radius) | (m > p2_mass)) & (d < 2 * p1_radius)
        smallidx = (d > 2 * p1_radius) & (~largeidx)
        
        m_rem = M - np.sum(m)
        
        if m_rem > M / 1000:
            if m_rem > (p2_mass - np.sum(m[smallidx])):
                rand_assign_frag = 1
            elif m_rem > (p1_mass - np.sum(m[largeidx])):
                rand_assign_frag = 2
            else:
                rand_assign_frag = 1 + int(round(np.random.rand()))
        
            if rand_assign_frag == 1:
                d_rem_approx = ((m_rem / p1_mass * p1_radius**3) ** (1/3)) * 2
                idx_rem1 = np.array([1])  # assign to larger satellite
                idx_rem2 = np.array([])
            else:
                d_rem_approx = ((m_rem / p2_mass * p2_radius**3) ** (1/3)) * 2
                idx_rem1 = np.array([])
                idx_rem2 = np.array([1])  # assign to smaller satellite
        
            Am_rem = func_Am(d_rem_approx, p1_objclass)
            A_rem = m_rem * Am_rem
            d_rem = d_rem_approx
        
            if (d_rem < LB).any() and (m_rem < M/1000):
                d_rem = np.array([])
                A_rem = np.array([])
                Am_rem = np.array([])
                m_rem = np.array([])
                idx_rem1 = np.array([])
                idx_rem2 = np.array([])
        else:
            d_rem = np.array([])
            A_rem = np.array([])
            Am_rem = np.array([])
            m_rem = np.array([])

    # -------------------------
    # Compute dv and dv_vec
    # -------------------------
    # In MATLAB, Am and Am_rem are concatenated and passed to func_dv.
    # If there are remnants, combine them; if not, use Am alone.
    if m_rem.size != 0:
        Am_all = np.concatenate((Am, Am_rem))
    else:
        Am_all = Am.copy()

    # Compute delta-v in m/s and convert to km/s.
    dv = func_dv(Am_all, 'col') / 1000  # dv is an array, in km/s

    # Generate random direction parameters (one per fragment)
    u = np.random.rand(len(dv)) * 2 - 1
    theta = np.random.rand(len(dv)) * 2 * np.pi

    v = np.sqrt(1 - u**2)
    # p is a (N x 3) matrix with directional cosines
    p = np.vstack((v * np.cos(theta), v * np.sin(theta), u)).T

    # Multiply each row by the corresponding dv (broadcasting)
    dv_vec = p * dv[:, np.newaxis]

    # -------------------------
    # Non-catastrophic case:
    # -------------------------
    if not isCatastrophic:
        # In MATLAB:
        # p1remnantmass = p1_mass + p2_mass - sum([m; m_rem]);
        # (note: [m; m_rem] is vertical concatenation)
        try:
            # Ensure m_rem is a 1D array
            m_rem = np.atleast_1d(m_rem)
        except Exception:
            m_rem = np.array([])

        p1remnantmass = p1_mass + p2_mass - np.sum(np.concatenate((m, m_rem)))  # could be larger than p1_mass
        # Append this remnant mass to m_rem:
        m_rem = np.concatenate((m_rem, [p1remnantmass]))
        # Append the corresponding approximate diameter, area and A/m
        d_rem = np.concatenate((d_rem, [p1_radius * 2]))
        A_rem = np.concatenate((A_rem, [np.pi * p1_radius**2]))  # area (for TLE generation, not used in func_create_tlesv2_vec)
        Am_rem = np.concatenate((Am_rem, [np.pi * p1_radius**2 / p1remnantmass]))  # A/m ratio for remnant

        # In MATLAB the parent object gets no extra delta-v, so append a zero.
        dv = np.concatenate((dv, [0]))
        # And append a zero row to dv_vec.
        dv_vec = np.vstack((dv_vec, [0, 0, 0]))

        # Now, total debris mass is the sum of (m and m_rem).
        total_debris_mass = np.sum(np.concatenate((m, m_rem)))
    else:
        # For the catastrophic case, ensure m_rem is an array.
        try:
            m_rem = np.atleast_1d(m_rem)
        except Exception:
            # If there are no remnants, return empty arrays (or handle as desired)
            fragments_all = np.array([])
            isCatastrophic = True
            # Optionally: return fragments_all, <other outputs>, isCatastrophic
            # For this snippet we simply exit.
            raise ValueError("No remnant fragments and catastrophic collision: nothing to output.")
        total_debris_mass = np.sum(np.concatenate((m, m_rem)))

    # -------------------------
    # Check overall mass consistency
    # -------------------------
    original_mass = p1_mass + p2_mass
    if abs(total_debris_mass - original_mass) > M * 0.05:
        print(f'Warning: Total sum of debris mass ({total_debris_mass:.1f} kg) differs from "mass" of original objects ({original_mass:.1f} kg)')
        # (In the MATLAB code, this warning uses the sum of p1 and p2 masses rather than M.)

    # -------------------------
    # Combine fragment properties
    # -------------------------
    # For fragments (the original ones) the MATLAB code builds:
    # fragments = [[d; d_rem] [A; A_rem] [Am; Am_rem] [m; m_rem] dv dv_vec(:,1) dv_vec(:,2) dv_vec(:,3)];
    #
    # That is, each row has:
    #   - Column 0: fragment diameter (d) 
    #   - Column 1: area (A)
    #   - Column 2: A/m
    #   - Column 3: mass (m)
    #   - Column 4: delta-v magnitude (dv)
    #   - Columns 5-7: the three components of dv_vec.
    #
    # In Python, we first vertically concatenate the original and remnant arrays.
    d_combined  = np.concatenate((d, d_rem))
    A_combined  = np.concatenate((A, A_rem))
    Am_combined = np.concatenate((Am, Am_rem))
    m_combined  = np.concatenate((m, m_rem))

    # Here we assume that dv and dv_vec have been computed for all fragments.
    # (If you computed dv (and hence dv_vec) based on the combined Am values then they already have the correct length.)
    # Otherwise, if dv and dv_vec were computed only for the original fragments, the non-catastrophic branch above has appended one extra row.
    # For clarity, we now build the final fragments array using the combined arrays.
    # Note: dv is a 1D array and dv_vec is a 2D array (with 3 columns).
    fragments_all = np.column_stack((d_combined,
                                    A_combined,
                                    Am_combined,
                                    m_combined,
                                    dv,            # delta-v magnitude
                                    dv_vec[:, 0],  # x component
                                    dv_vec[:, 1],  # y component
                                    dv_vec[:, 2])) # z component

    # fragments_all now has one row per fragment, with 8 columns.`
        
    # Distribute fragments amongst parent 1 and parent 2
    if isCatastrophic:
        # Initialize assignment arrays based on fragment diameters.
        largeidx_all = (fragments_all[:, 0] > 2 * p2_radius) & (fragments_all[:, 0] < 2 * p1_radius)
        smallidx_all = (fragments_all[:, 0] > 2 * p1_radius) & (~largeidx_all)
        
        # If remnant indices exist, include them in the assignment.
        if 'idx_rem1' in locals() and idx_rem1.size > 0:
            largeidx_all = largeidx_all | np.isin(np.arange(len(fragments_all)), idx_rem1)
        if 'idx_rem2' in locals() and idx_rem2.size > 0:
            smallidx_all = smallidx_all | np.isin(np.arange(len(fragments_all)), idx_rem2)
        
        assignedidx = largeidx_all | smallidx_all
        idx_unassigned = np.where(~assignedidx)[0]
        
        # Initially assign fragments according to the indices.
        fragments1 = fragments_all[largeidx_all, :]
        fragments2 = fragments_all[smallidx_all, :]
        
        if idx_unassigned.size > 0:
            fragments_unassigned = fragments_all[idx_unassigned, :]
            cum_mass_p1 = np.sum(fragments1[:, 3])
            cum_mass_p1 += np.cumsum(fragments_unassigned[:, 3])
            # p1_assign is a boolean array: True if cumulative mass is still below p1_mass.
            p1_assign = cum_mass_p1 <= p1_mass
            p1indx = np.where(p1_assign)[0]
            # Add the unassigned fragments to fragments1 (up to p1indx) and the rest to fragments2.
            fragments1 = np.vstack([fragments1, fragments_unassigned[:len(p1indx), :]])
            fragments2 = np.vstack([fragments2, fragments_unassigned[len(p1indx):, :]])
    else:
        # Non-catastrophic collision: assign the heaviest fragment to the large object.
        if fragments_all.shape[0] > 0:
            heaviestInd = fragments_all[:, 3] == np.max(fragments_all[:, 3])
            lighterInd = ~heaviestInd
            fragments1 = fragments_all[heaviestInd, :]
            fragments2 = fragments_all[lighterInd, :]
        else:
            fragments1 = np.array([])
            fragments2 = np.array([])

    # Remove fragments whose diameters are below LB.
    if fragments1.size > 0:
        fragments1 = fragments1[fragments1[:, 0] >= LB, :]
    if fragments2.size > 0:
        fragments2 = fragments2[fragments2[:, 0] >= LB, :]

    # Create debris objects.
    debris1 = func_create_tlesv2_vec(ep, p1_r, p1_v, p1_objclass, fragments1, param)
    param['maxID'] += debris1.shape[0]

    debris2 = func_create_tlesv2_vec(ep, p2_r, p2_v, p2_objclass, fragments2, param)
    param['maxID'] += debris2.shape[0]

    # Return debris objects and the collision type flag.

    if orbit_type == 'LEO':
        # Then remove all of the fragments that are in LEO and larger than minimum_altitude
        debris1 = debris1[(debris1[:, 0] < 8371) & (debris1[:, 0] > minimum_altitude), :]
        debris2 = debris2[(debris2[:, 0] < 8371) & (debris2[:, 0] > minimum_altitude), :]
    # Also check that no debris has a eccentricity greater than 1
    if debris1.size > 0:
        debris1 = debris1[debris1[:, 1] < 1, :]

    if debris2.size > 0:
        debris2 = debris2[debris2[:, 1] < 1, :]

    return debris1, debris2, isCatastrophic

def func_create_tlesv2_vec(ep, r_parent, v_parent, class_parent, fragments, param):
    """
    Create new satellite objects from fragmentation information.

    Parameters:
    - ep: Epoch
    - r_parent: Parent satellite position vector [x, y, z]
    - v_parent: Parent satellite velocity vector [vx, vy, vz]
    - class_parent: Parent satellite object class
    - fragments: Nx8 array containing fragment data:
        [diameter, Area, AMR, mass, total_dv, dv_X, dv_Y, dv_Z]
    - param: Dictionary containing parameters like 'max_frag', 'mu', 'req', 'maxID'

    Returns:
    - mat_frag: Array with fragment orbital elements and related data
    """

    max_frag = param.get('max_frag', float('inf'))
    mu = param.get('mu')
    req = param.get('req')
    maxID = param.get('maxID', 0)

    r0 = np.array(r_parent)
    v0 = np.array(v_parent)

    num_fragments = fragments.shape[0]

    if num_fragments > max_frag:
        print(f"Warning: number of fragments {num_fragments} exceeds max_frag {max_frag}")
    n_frag = min(num_fragments, max_frag)

    # Sort fragments by mass in descending order
    sort_idx = np.argsort(-fragments[:, 3])  # mass is in column 4 (index 3)
    fragments = fragments[sort_idx[:n_frag], :]

    # Compute new velocities by adding dv to parent velocity
    v = np.column_stack((
        fragments[:, 5] + v0[0],
        fragments[:, 6] + v0[1],
        fragments[:, 7] + v0[2]
    ))

    # Positions are the same as parent position
    r = np.tile(r0, (n_frag, 1))

    # Compute orbital elements from position and velocity vectors
    r = np.atleast_2d(r)
    v = np.atleast_2d(v)

    # Initialize lists to store orbital elements
    a_list = []
    ecc_list = []
    incl_list = []
    nodeo_list = []
    argpo_list = []
    mo_list = []

    # mu in poliastro terms
    k = Earth.k.to_value(u.km ** 3 / u.s ** 2)
    
    for i in range(n_frag):
        p, ecc, incl, nodeo, argpo, mo = rv2coe(k, r[i], v[i]) # gives semi-latus rectum not sma. 
        a = p / (1 - ecc ** 2) # Convert semi-latus rectum (p) to semi-major axis (a)
        a_list.append(a)
        ecc_list.append(ecc)
        incl_list.append(incl)
        nodeo_list.append(nodeo)
        argpo_list.append(argpo)
        mo_list.append(mo)

    # Convert lists to arrays
    a = np.array(a_list)
    ecc = np.array(ecc_list)
    incl = np.array(incl_list)
    nodeo = np.array(nodeo_list)
    argpo = np.array(argpo_list)
    mo = np.array(mo_list)

    # Process only valid orbits (a > 6371)
    idx_a = np.where(a > 6378)[0]
    num_a = len(idx_a)

    a = a[idx_a] 
    ecco = ecc[idx_a]
    inclo = incl[idx_a]
    nodeo = nodeo[idx_a]
    argpo = argpo[idx_a]
    mo = mo[idx_a]

    # Bstar parameter
    rho_0 = 0.157  # kg/(m^2 * Re)
    A_M = fragments[idx_a, 2]  # AMR is in column 3 (index 2)
    bstar = (0.5 * 2.2 * rho_0) * A_M  # Bstar in units of 1/Re

    mass = fragments[idx_a, 3]  # mass is in column 4 (index 3)
    radius = fragments[idx_a, 0] / 2  # diameter is in column 1 (index 0)

    errors = np.zeros(num_a)
    controlled = np.zeros(num_a)
    a_desired = np.full(num_a, np.nan)
    missionlife = np.full(num_a, np.nan)
    constel = np.zeros(num_a)

    date_created = np.full(num_a, ep)
    launch_date = np.full(num_a, np.nan)

    frag_objectclass = np.full(num_a, filter_objclass_fragments_int(class_parent))

    ID_frag = np.arange(maxID + 1, maxID + num_a + 1)

    # Assemble mat_frag
    mat_frag = np.column_stack((
        a, ecco, inclo, nodeo, argpo, mo, bstar, mass, radius,
        errors, controlled, a_desired, missionlife, constel,
        date_created, launch_date, r[idx_a], v[idx_a], frag_objectclass, ID_frag
    ))

    return mat_frag


def filter_objclass_fragments_int(class_parent):
    """
    Assign object class to fragments according to the parent particle.

    For this implementation, we'll assume fragments inherit the parent's class.
    """
    return class_parent

# Example usage
if __name__ == "__main__":
    # Define p1_in and p2_in
    # p1_in = 1.0e+03 * np.array([1.2500, 0.0040, 2.8016, 2.7285, 6.2154, -0.0055, -0.0030, 0.0038, 0.0010])

    # p2_in = 1.0e+03 * np.array([0.0060, 0.0001, 2.8724, 2.7431, 6.2248, 0.0032, 0.0054, -0.0039, 0.0010])

    p1_in = np.array([
        1000,  # mass in kg
        5,     # radius in meters
        2100.4,  # r_x in km
        2100.1,  # r_y in km
        6224.8,  # r_z in km
        -5.5,    # v_x in km/s
        -3.0,    # v_y in km/s
        3.8,     # v_z in km/s
        1.0      # object_class (dimensionless)
    ])

    p2_in = np.array([
        100,     # mass in kg
        1,     # radius in meters
        2100.4,  # r_x in km
        2100.1,  # r_y in km
        6224.8,  # r_z in km
        3.2,     # v_x in km/s
        5.4,     # v_y in km/s
        -3.9,    # v_z in km/s
        1.0      # object_class (dimensionless)
    ])
    
    # Define the param dictionary
    param = {
        'req': 6.3781e+03,
        'mu': 3.9860e+05,
        'j2': 0.0011,
        'max_frag': float('inf'),  # Inf in MATLAB translates to float('inf') in Python
        'maxID': 0,
        'density_profile': 'static'
    }

    # Lower bound (LB)
    LB = 0.1  # Assuming this is the lower bound in meters

    debris1, debris2, isCatastrophic = frag_col_SBM_vec_lc2(0, p1_in, p2_in, param, LB)

    # if debris 1 is empty then contine
    if debris1.size == 0:
        print("No debris fragments were generated")
        exit()

    # Assuming debris1 is already defined
    idx_a = 0
    idx_ecco = 1

    # 1. 1D Histogram for SMA (semi-major axis)
    plt.figure()
    plt.hist((debris1[:, idx_a] - 1) * 6378, bins=np.arange(0, 5001, 100))
    plt.title('SMA as altitude (km)')
    plt.xlabel('SMA as altitude (km)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # 2. 1D Histogram for Eccentricity
    plt.figure()
    plt.hist(debris1[:, idx_ecco], bins=50)
    plt.title('Eccentricity')
    plt.xlabel('Eccentricity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # 3. 2D Histogram using histogram2d with LogNorm for color scale
    plt.figure()
    hist, xedges, yedges = np.histogram2d(
        (debris1[:, idx_a] - 1) * 6371, debris1[:, idx_ecco],
        bins=[np.arange(0, 5001, 100), np.arange(0, 1.01, 0.01)]
    )

    # Avoid any zero counts for logarithmic color scaling
    hist[hist == 0] = np.nan  # Replace zeros with NaNs to avoid LogNorm issues

    # Plotting the 2D histogram
    mappable = plt.imshow(
        hist.T, origin='lower', norm=LogNorm(), 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto'
    )
    plt.colorbar(mappable, label='Count')
    plt.xlim([0, 3000])
    plt.xlabel('SMA as altitude (km)')
    plt.ylabel('Eccentricity')
    plt.title('2D Histogram of SMA and Eccentricity')
    plt.grid(True)
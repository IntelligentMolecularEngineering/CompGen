import torch
import numpy as np


def mixing_rule_density(mw_i, vm_298k_i, x_volume, T=293.15, device='cuda'):
    """
    Calculate density at temperature T based on <Characterization and
    Properties of Petroleum Fractions> Eq 2.110.

    Parameters:
    mw_i (torch.Tensor): Molecular weight (g/mol)
    vm_298k_i (torch.Tensor): Molar volume at 298K (cc/mol)
    x_volume (torch.Tensor): Volume fraction
    T (float): Temperature (K)

    Returns:
    torch.Tensor: Density (g/cm^3)
    """
    mw_i = mw_i.to(device)
    vm_298k_i = vm_298k_i.to(device)
    x_volume = x_volume.to(device)

    # Constants
    T0 = 298.0

    # Density function parameters
    density298k_i = mw_i / vm_298k_i
    densityT_i = (density298k_i - 2.34e-3 * (T - T0)) / (1 - 1.9e-3 * (T - T0))
    density = torch.sum(densityT_i * x_volume, dim=1)

    return density


def mixing_rule_element_content_wt(C_num_i, H_num_i, S_num_i, N_num_i, O_num_i, mw_i, x_mass, device='cuda'):
    """
    Calculate the element content in weight percent based on mass fraction.

    Args:
        C_num_i (torch.Tensor): Carbon number (molecular formula count).
        H_num_i (torch.Tensor): Hydrogen number.
        S_num_i (torch.Tensor): Sulfur number.
        N_num_i (torch.Tensor): Nitrogen number.
        O_num_i (torch.Tensor): Oxygen number.
        mw_i (torch.Tensor): Molecular weight in g/mol.
        x_mass (torch.Tensor): Mass fraction of each component.
        device (str): The device to run the calculations on, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
        float: Weight percent content of C, H, S, N, and O.
    """

    # Move tensors to the specified device (GPU or CPU)
    C_num_i = C_num_i.to(device)
    H_num_i = H_num_i.to(device)
    S_num_i = S_num_i.to(device)
    N_num_i = N_num_i.to(device)
    O_num_i = O_num_i.to(device)
    mw_i = mw_i.to(device)
    x_mass = x_mass.to(device)

    # Calculate content of each element (in weight percent)
    C_content_i = 12.0107 * C_num_i / mw_i
    H_content_i = 1.008 * H_num_i / mw_i
    S_content_i = 32.065 * S_num_i / mw_i
    N_content_i = 14.0067 * N_num_i / mw_i
    O_content_i = 15.9994 * O_num_i / mw_i

    # Calculate weighted sum of each element's content
    C_content = torch.sum(C_content_i * x_mass, dim=1) * 100
    H_content = torch.sum(H_content_i * x_mass, dim=1) * 100
    S_content = torch.sum(S_content_i * x_mass, dim=1) * 100
    N_content = torch.sum(N_content_i * x_mass, dim=1) * 100
    O_content = torch.sum(O_content_i * x_mass, dim=1) * 100

    # Normalize the contents so that they sum to 100%
    total_content = C_content + H_content + S_content + N_content + O_content

    # Normalize each element's content
    C_content = (C_content / total_content) * 100
    H_content = (H_content / total_content) * 100
    S_content = (S_content / total_content) * 100
    N_content = (N_content / total_content) * 100
    O_content = (O_content / total_content) * 100

    return C_content, H_content, S_content, N_content, O_content


def mixing_rule_GC_MS_type_diesel(S_num_i, N_num_i, O_num_i, aro_ring_num_i, naph_ring_num_i, x_mass, device='cuda'):
    # Ensure inputs are tensors and moved to the appropriate device
    S_num_i = S_num_i.to(device)
    N_num_i = N_num_i.to(device)
    O_num_i = O_num_i.to(device)
    aro_ring_num_i = aro_ring_num_i.to(device)
    naph_ring_num_i = naph_ring_num_i.to(device)
    x_mass = x_mass.to(device)

    # Determine alkane assignment
    is_alkane = (aro_ring_num_i == 0) & (naph_ring_num_i == 0) & (S_num_i == 0) & (N_num_i == 0) & (O_num_i == 0)
    alkane_weight_fraction = torch.sum(x_mass * is_alkane, dim=1)

    # Determine naphthene assignments
    is_one_ring_naphthene = (aro_ring_num_i == 0) & (naph_ring_num_i == 1) & (S_num_i == 0) & (N_num_i == 0) & (
            O_num_i == 0)
    one_ring_naphthene_weight_fraction = torch.sum(x_mass * is_one_ring_naphthene, dim=1)
    is_two_ring_naphthene = (aro_ring_num_i == 0) & (naph_ring_num_i == 2) & (S_num_i == 0) & (N_num_i == 0) & (
            O_num_i == 0)
    two_ring_naphthene_weight_fraction = torch.sum(x_mass * is_two_ring_naphthene, dim=1)
    is_three_ring_naphthene = (aro_ring_num_i == 0) & (naph_ring_num_i == 3) & (S_num_i == 0) & (N_num_i == 0) & (
            O_num_i == 0)
    three_ring_naphthene_weight_fraction = torch.sum(x_mass * is_three_ring_naphthene, dim=1)
    is_four_ring_naphthene = (aro_ring_num_i == 0) & (naph_ring_num_i == 4) & (S_num_i == 0) & (N_num_i == 0) & (
            O_num_i == 0)
    four_ring_naphthene_weight_fraction = torch.sum(x_mass * is_four_ring_naphthene, dim=1)
    is_five_ring_naphthene = (aro_ring_num_i == 0) & (naph_ring_num_i == 5) & (S_num_i == 0) & (N_num_i == 0) & (
            O_num_i == 0)
    five_ring_naphthene_weight_fraction = torch.sum(x_mass * is_five_ring_naphthene, dim=1)
    is_six_and_more_ring_naphthene = (aro_ring_num_i == 0) & (naph_ring_num_i > 5) & (S_num_i == 0) & (N_num_i == 0) & (
            O_num_i == 0)
    six_and_more_ring_naphthene_weight_fraction = torch.sum(x_mass * is_six_and_more_ring_naphthene, dim=1)

    # Determine aromatic hydrocarbon assignments
    is_one_ring_aromatic = (aro_ring_num_i == 1) & (S_num_i == 0) & (N_num_i == 0) & (O_num_i == 0)
    one_ring_aromatic_weight_fraction = torch.sum(x_mass * is_one_ring_aromatic, dim=1)
    is_two_ring_aromatic = (aro_ring_num_i == 2) & (S_num_i == 0) & (N_num_i == 0) & (O_num_i == 0)
    two_ring_aromatic_weight_fraction = torch.sum(x_mass * is_two_ring_aromatic, dim=1)
    is_three_ring_aromatic = (aro_ring_num_i == 3) & (S_num_i == 0) & (N_num_i == 0) & (O_num_i == 0)
    three_ring_aromatic_weight_fraction = torch.sum(x_mass * is_three_ring_aromatic, dim=1)
    is_four_ring_aromatic = (aro_ring_num_i == 4) & (S_num_i == 0) & (N_num_i == 0) & (O_num_i == 0)
    four_ring_aromatic_weight_fraction = torch.sum(x_mass * is_four_ring_aromatic, dim=1)
    is_five_ring_aromatic = (aro_ring_num_i == 5) & (S_num_i == 0) & (N_num_i == 0) & (O_num_i == 0)
    five_ring_aromatic_weight_fraction = torch.sum(x_mass * is_five_ring_aromatic, dim=1)
    is_six_and_more_ring_aromatic = (aro_ring_num_i > 5) & (S_num_i == 0) & (N_num_i == 0) & (O_num_i == 0)
    six_and_more_ring_aromatic_weight_fraction = torch.sum(x_mass * is_six_and_more_ring_aromatic, dim=1)

    # Determine sulfur-containing compound assignments
    is_2sulfur = (aro_ring_num_i == 0) & (naph_ring_num_i == 2) & (S_num_i == 1) & (N_num_i == 0) & (O_num_i == 0)
    two_ring_naphthene_weight_fraction += torch.sum(x_mass * is_2sulfur, dim=1)
    is_3sulfur = (aro_ring_num_i == 0) & (naph_ring_num_i == 3) & (S_num_i == 1) & (N_num_i == 0) & (O_num_i == 0)
    three_ring_naphthene_weight_fraction += torch.sum(x_mass * is_3sulfur, dim=1)
    is_benzothiophene = (aro_ring_num_i >= 1) & (aro_ring_num_i <= 2) & (S_num_i == 1) & (N_num_i == 0) & (O_num_i == 0)
    one_ring_aromatic_weight_fraction += torch.sum(x_mass * is_benzothiophene, dim=1)
    is_dibenzothiophene = (aro_ring_num_i == 3) & (S_num_i == 1) & (N_num_i == 0) & (O_num_i == 0)
    two_ring_aromatic_weight_fraction += torch.sum(x_mass * is_dibenzothiophene, dim=1)
    is_0sulfur = (aro_ring_num_i == 0) & (naph_ring_num_i == 0) & (S_num_i == 1) & (N_num_i == 0) & (O_num_i == 0)
    alkane_weight_fraction += torch.sum(x_mass * is_0sulfur, dim=1)
    is_1sulfur = (aro_ring_num_i == 0) & (naph_ring_num_i == 1) & (S_num_i == 1) & (N_num_i == 0) & (O_num_i == 0)
    one_ring_naphthene_weight_fraction += torch.sum(x_mass * is_1sulfur, dim=1)

    # Determine nitrogen-containing compound assignments
    is_2amine = (aro_ring_num_i == 0) & (naph_ring_num_i == 2) & (S_num_i == 0) & (N_num_i == 1) & (O_num_i == 0)
    two_ring_naphthene_weight_fraction += torch.sum(x_mass * is_2amine, dim=1)
    is_3amine = (aro_ring_num_i == 0) & (naph_ring_num_i == 3) & (S_num_i == 0) & (N_num_i == 1) & (O_num_i == 0)
    three_ring_naphthene_weight_fraction += torch.sum(x_mass * is_3amine, dim=1)
    is_indole = (aro_ring_num_i == 1) & (S_num_i == 0) & (N_num_i == 1) & (O_num_i == 0)
    one_ring_aromatic_weight_fraction += torch.sum(x_mass * is_indole, dim=1)
    is_carbazole = (aro_ring_num_i == 2) & (S_num_i == 0) & (N_num_i == 1) & (O_num_i == 0)
    two_ring_aromatic_weight_fraction += torch.sum(x_mass * is_carbazole, dim=1)
    is_pyrrole = (aro_ring_num_i == 0) & (naph_ring_num_i == 1) & (S_num_i == 0) & (N_num_i == 1) & (O_num_i == 0)
    one_ring_aromatic_weight_fraction += torch.sum(x_mass * is_pyrrole, dim=1)
    is_carbazole3 = (aro_ring_num_i == 3) & (S_num_i == 0) & (N_num_i == 1) & (O_num_i == 0)
    three_ring_aromatic_weight_fraction += torch.sum(x_mass * is_carbazole3, dim=1)

    # Return the individual weight fractions
    return (alkane_weight_fraction,
            one_ring_naphthene_weight_fraction,
            two_ring_naphthene_weight_fraction,
            three_ring_naphthene_weight_fraction,
            four_ring_naphthene_weight_fraction,
            five_ring_naphthene_weight_fraction,
            six_and_more_ring_naphthene_weight_fraction,
            one_ring_aromatic_weight_fraction,
            two_ring_aromatic_weight_fraction,
            three_ring_aromatic_weight_fraction,
            four_ring_aromatic_weight_fraction,
            five_ring_aromatic_weight_fraction,
            six_and_more_ring_aromatic_weight_fraction)


def mixing_rule_distillation_cum_fraction_d86(x_v, tb_i, device='cuda'):
    """
    Calculate ASTM D86 cumulative fraction distillation curve by linear interpolation using PyTorch.

    Args:
        x_v (torch.Tensor): Fraction (1D tensor).
        tb_i (torch.Tensor): Boiling temperatures (1D tensor) in Kelvin.
        device (str): The device to run on, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
        torch.Tensor: ASTM D86 distillation profile in Kelvin.
    """

    # Move tensors to the specified device (GPU or CPU)
    x_v = x_v.to(device)
    tb_i = tb_i.to(device)

    # Remove zero entries from x_v and tb_i
    zeroindex = (x_v == 0)
    mask = ~zeroindex  # Boolean mask where True means non-zero
    masked_x_v = x_v * mask  # Element-wise multiplication to keep non-zero entries
    masked_tb_i = tb_i * mask  # Element-wise multiplication to keep non-zero entries

    # Combine x_v and tb_i into a 2D tensor for sorting
    br = torch.stack((masked_x_v, masked_tb_i), dim=2)
    sorted_indices = torch.argsort(br[:, :, 1], dim=-1)
    newbr = br.gather(-2, sorted_indices.unsqueeze(-1).expand(-1, -1, br.size(-1)))

    sorted_x_v = newbr[:, :, 0]
    sorted_tb_i = newbr[:, :, 1]

    # Compute cumulative sum of the sorted fractions
    sorted_x_cv = torch.cumsum(sorted_x_v, dim=-1)

    # Predefined target cumulative fractions (as a tensor)
    all_point = torch.tensor([0.005, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                              0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.995], device=device)
    Pre_pred = torch.zeros(all_point.size(0), x_v.shape[0], device=device)  # Initialize the prediction tensor

    # Perform linear interpolation for each target cumulative fraction
    for i in range(all_point.size(0)):  # Iterate over the target cumulative fractions
        vol = all_point[i]

        for j in range(sorted_x_cv.size(0)):  # Iterate over each row in sorted_x_cv and sorted_tb_i
            # Find the indices where cumulative sum is less than or equal to vol for the j-th row
            xfront = sorted_x_cv[j, sorted_x_cv[j] <= vol]  # Apply condition on the j-th row
            yfront = sorted_tb_i[j, sorted_x_cv[j] <= vol]  # Apply condition on the j-th row
            xback = sorted_x_cv[j, sorted_x_cv[j] > vol]  # Apply condition on the j-th row
            yback = sorted_tb_i[j, sorted_x_cv[j] > vol]  # Apply condition on the j-th row

            if xfront.numel() == 0:
                xf = xback[0]
                yf = yback[0]
                xb = xback[0] - 1e-9
                yb = yback[0] - 1e-9
            else:
                xf = xfront[-1]
                yf = yfront[-1]
                xb = xback[0]
                yb = yback[0]

            # Linear interpolation formula for each row and each target cumulative fraction
            Pre_pred[i, j] = yf + (vol - xf) * (yb - yf) / (xb - xf)

    # Convert the simulated distillation profile to ASTM D86 by CG_convert_sd_to_d86 function
    distillation_cum_fraction_d86_v = CG_convert_sd_to_d86(all_point, Pre_pred,
                                                           all_point) + 273.15  # Convert to absolute temperature
    distillation_cum_fraction_d86_v = torch.abs(distillation_cum_fraction_d86_v)  # Ensure positive values

    return distillation_cum_fraction_d86_v


def CG_convert_sd_to_d86(Pre_Point, Pre_pred, T_range_vol_exp, device='cuda'):
    """
    Convert simulated distillation (SD) data to ASTM D86 using linear regression and predefined constants.

    Args:
        Pre_Point (torch.Tensor): Predefined points for interpolation (1D tensor).
        Pre_pred (torch.Tensor): Predicted values corresponding to Pre_Point (1D tensor).
        T_range_vol_exp (torch.Tensor): Experimental volume fractions (1D tensor).
        device (str): The device to run on, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
        torch.Tensor: The ASTM D86 distillation curve in Celsius.
    """

    # Move tensors to the specified device (GPU or CPU)
    Pre_Point = Pre_Point.to(device)
    Pre_pred = Pre_pred.to(device)
    T_range_vol_exp = T_range_vol_exp.to(device)

    # Convert from Kelvin to Fahrenheit (Pre_pred is assumed to be in Kelvin)
    Pre_pred_F = Pre_pred * 1.8 - 459.67

    # Constants for linear regression
    E = torch.tensor([2.6029, 0.30785, 0.14862, 0.07978, 0.06069, 0.30470], device=device)
    F = torch.tensor([0.6596, 1.2341, 1.4287, 1.5386, 1.5176, 1.1259], device=device)

    # Find specific SD values at key points
    SD_50 = Pre_pred_F[Pre_Point == 0.5]
    SD_0 = Pre_pred_F[Pre_Point == 0.005]
    SD_10 = Pre_pred_F[Pre_Point == 0.1]
    SD_30 = Pre_pred_F[Pre_Point == 0.3]
    SD_70 = Pre_pred_F[Pre_Point == 0.7]
    SD_90 = Pre_pred_F[Pre_Point == 0.9]
    SD_100 = Pre_pred_F[Pre_Point == 0.995]

    # Calculate U values using the differences between SD values
    U1 = E[0] * (SD_100 - SD_90) ** F[0]
    U2 = E[1] * (SD_90 - SD_70) ** F[1]
    U3 = E[2] * (SD_70 - SD_50) ** F[2]
    U4 = E[3] * (SD_50 - SD_30) ** F[3]
    U5 = E[4] * (SD_30 - SD_10) ** F[4]
    U6 = E[5] * (SD_10 - SD_0) ** F[5]

    # Calculate D86_50 (starting point)
    D86_50 = 0.77601 * (SD_50) ** 1.0395

    # Initialize output tensor for D86 values
    D86 = torch.zeros_like(Pre_pred, device=device)

    # Loop over experimental volume fractions and apply the conversion rules
    for i in range(len(T_range_vol_exp)):
        vol = T_range_vol_exp[i]

        if vol < 0.1 and vol >= 0.005:
            Ui = E[5] * (SD_10 - Pre_pred_F[Pre_Point == vol]) ** F[5]
            D86[i, :] = D86_50 - U4 - U5 - Ui
        elif vol < 0.3 and vol >= 0.1:
            Ui = E[4] * (SD_30 - Pre_pred_F[Pre_Point == vol]) ** F[4]
            D86[i, :] = D86_50 - U4 - Ui
        elif vol < 0.5 and vol >= 0.3:
            Ui = E[3] * (SD_50 - Pre_pred_F[Pre_Point == vol]) ** F[3]
            D86[i, :] = D86_50 - Ui
        elif vol == 0.5:
            D86[i, :] = D86_50
        elif vol > 0.5 and vol <= 0.7:
            Ui = E[2] * (Pre_pred_F[Pre_Point == vol] - SD_50) ** F[2]
            D86[i, :] = D86_50 + Ui
        elif vol > 0.7 and vol <= 0.9:
            Ui = E[1] * (Pre_pred_F[Pre_Point == vol] - SD_70) ** F[1]
            D86[i, :] = D86_50 + U3 + Ui
        elif vol > 0.9 and vol <= 0.995:
            Ui = E[0] * (Pre_pred_F[Pre_Point == vol] - SD_90) ** F[0]
            D86[i, :] = D86_50 + U3 + U2 + Ui
        else:
            print("PredPoint is not in the range")

    # Convert from Fahrenheit to Celsius
    D86 = (D86 - 32) * 5 / 9

    return D86


def process_mol_data(mol_index, mol_matrix, device='cuda'):
    """
    Process molecular index and matrix data using PyTorch, running computations on the GPU.

    Args:
        mol_index (torch.Tensor): Tensor containing the molecular index.
        mol_matrix (torch.Tensor): Tensor containing the molecular matrix.
        device (str): Device to perform calculations ('cuda' or 'cpu'). Default is 'cuda'.

    Returns:
        torch.Tensor: Processed mol_content tensor.
    """
    # Ensure tensors are moved to the specified device
    mol_index = mol_index.to(device).squeeze(1)
    mol_matrix = mol_matrix.to(device).squeeze(1)

    # Get the shape of the tensors to identify the last two dimensions
    *batch_shape, index_size, matrix_size = mol_index.shape

    # Process the last two dimensions (index and matrix)
    zero_index = (mol_index == 0)
    mol_matrix[zero_index] = 0.0
    mol_matrix = abs(mol_matrix)
    mol_matrix = mol_matrix / torch.sum(mol_matrix, dim=(-2, -1), keepdim=True)

    # Flatten the last two dimensions (index and matrix)
    mol_index_flat = mol_index.flatten(start_dim=-2)
    mol_matrix_flat = mol_matrix.flatten(start_dim=-2)

    # Combine the flattened tensors
    combined_matrix = torch.stack((mol_index_flat, mol_matrix_flat), dim=-1)

    # Sort the rows by the first column (mol_index_flat) for the last dimension
    sorted_indices = torch.argsort(combined_matrix[:, :, 0], dim=-1)
    sorted_combined_matrix = combined_matrix.gather(-2, sorted_indices.unsqueeze(-1).expand(-1, -1, combined_matrix.size(-1)))

    # Remove rows where all elements are zero
    all_zero_rows = torch.all(sorted_combined_matrix == 0, dim=-1)
    sorted_combined_matrix = sorted_combined_matrix[~all_zero_rows]

    # Reorganize the tensor shape back
    sorted_combined_matrix = sorted_combined_matrix.view(*batch_shape, -1, 2)

    # Extract the second column (mol_content) which corresponds to mol_matrix
    mol_content = sorted_combined_matrix[..., 1]

    return mol_content


def mixing_rule_cetane_number(cn_beta_array, cn_array, x_volume, device='cuda'):
    """
    Calculate bulk cetane number (CN) based on Ghosh model equation.

    Parameters:
    cn_beta_array (torch.Tensor): CN beta parameters for each component
    cn_array (torch.Tensor): Cetane number for each component
    x_volume (torch.Tensor): Volume fraction
    device (str): Device to use ('cuda' or 'cpu')
    eps (float): Small value to prevent division by zero

    Returns:
    torch.Tensor: Bulk cetane number
    """
    # Move tensors to specified device
    cn_beta_array = cn_beta_array.to(device)
    cn_array = cn_array.to(device)
    x_volume = x_volume.to(device)

    # Calculate numerator: sum(x_volume * CN_beta * CN)
    numerator = torch.sum(x_volume * cn_beta_array * cn_array, dim=1)

    # Calculate denominator: sum(x_volume * CN_beta)
    denominator = torch.sum(x_volume * cn_beta_array, dim=1)

    # Calculate bulk cetane number with numerical stability
    cn_bulk = numerator / (denominator)

    return cn_bulk


def mixing_rule_freeze_point(tm_i, x_molar, device='cuda'):
    """
    Calculate bulk freeze point based on molar fraction weighted average.

    Parameters:
    tm_i (torch.Tensor): Freeze point of each component (K)
    x_molar (torch.Tensor): Molar fraction
    device (str): Device to use ('cuda' or 'cpu')

    Returns:
    torch.Tensor: Bulk freeze point (K)
    """
    # Move tensors to specified device
    tm_i = tm_i.to(device)
    x_molar = x_molar.to(device)

    # Calculate bulk freeze point: sum(tm_i * x_molar)
    freeze_point = torch.sum(tm_i * x_molar, dim=1)

    return freeze_point


def mixing_rule_gasoline_PIONA(mass_fraction, PIONA, device='cuda'):
    """
    Calculate normalized mass fractions for PIONA categories.

    Parameters:
    mass_fraction (torch.Tensor): Mass fraction of each component, shape (batch_size, num_components)
    PIONA (torch.Tensor): PIONA category indices (1=P, 2=I, 3=O, 4=N, 5=A), shape (batch_size, num_components)
    mw (torch.Tensor): Molecular weights (unused in calculation)
    device (str): Device to use ('cuda' or 'cpu')

    Returns:
    torch.Tensor: Normalized mass fractions for each PIONA category, shape (batch_size, 5)
    """
    # Move tensors to specified device
    mass_fraction = mass_fraction.to(device)
    PIONA = PIONA.to(device)

    # Normalize mass fractions per sample
    normalized_mass = mass_fraction / torch.sum(mass_fraction, dim=1, keepdim=True)

    # Create category masks
    p_mask = (PIONA == 1).float()
    i_mask = (PIONA == 2).float()
    o_mask = (PIONA == 3).float()
    n_mask = (PIONA == 4).float()
    a_mask = (PIONA == 5).float()

    # Calculate category mass fractions
    gasoline_P_mass_fraction = torch.sum(normalized_mass * p_mask, dim=1)
    gasoline_I_mass_fraction = torch.sum(normalized_mass * i_mask, dim=1)
    gasoline_O_mass_fraction = torch.sum(normalized_mass * o_mask, dim=1)
    gasoline_N_mass_fraction = torch.sum(normalized_mass * n_mask, dim=1)
    gasoline_A_mass_fraction = torch.sum(normalized_mass * a_mask, dim=1)

    # Stack and normalize categories
    # category_masses = torch.stack((gasoline_P_mass_fraction, i_mass, o_mass, n_mass, a_mass), dim=1)
    # normalized_categories = category_masses / torch.sum(category_masses, dim=1, keepdim=True)

    return (gasoline_P_mass_fraction,
            gasoline_I_mass_fraction,
            gasoline_O_mass_fraction,
            gasoline_N_mass_fraction,
            gasoline_A_mass_fraction)

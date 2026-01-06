import torch


def fraction_conversion(mw, vm_298, in_fraction_value, in_fraction_type, out_fraction_type, device='cpu'):
    """
    Convert fraction based on input molecular property table using PyTorch.

    Args:
        mol_prop_tensor: [dict] A dictionary with molecular properties, including 'mw' (molecular weight) and 'vm_298K' (molar volume at 298K)
        in_fraction_value: [torch.Tensor] Input fraction values (1D tensor) that sum to 1
        in_fraction_type: [str] ('molar' | 'mass' | 'volume') The input fraction type
        out_fraction_type: [str] ('molar' | 'mass' | 'volume') The desired output fraction type
        device: [str] The device ('cpu' or 'cuda') where the computation should occur.

    Returns:
        out_fraction_value: [torch.Tensor] The output fraction values (1D tensor) that sum to 1.
    """
    # Move inputs to the correct device (GPU or CPU)
    in_fraction_value = in_fraction_value.to(device)
    mw = mw.to(device)
    vm_298K = vm_298.to(device)

    # Conversion logic
    if in_fraction_type == 'molar':
        if out_fraction_type == 'mass':
            # Convert from molar fraction to mass fraction
            temp_fraction = in_fraction_value * mw
            out_fraction_value = temp_fraction / temp_fraction.sum(dim=-1, keepdim=True)
            return out_fraction_value

        elif out_fraction_type == 'volume':
            # Convert from molar fraction to volume fraction
            temp_fraction = in_fraction_value * vm_298K
            out_fraction_value = temp_fraction / temp_fraction.sum(dim=-1, keepdim=True)
            return out_fraction_value

    elif in_fraction_type == 'mass':
        if out_fraction_type == 'molar':
            # Convert from mass fraction to molar fraction
            temp_fraction = in_fraction_value / mw
            out_fraction_value = temp_fraction / temp_fraction.sum(dim=-1, keepdim=True)
            return out_fraction_value

        elif out_fraction_type == 'volume':
            # Convert from mass fraction to volume fraction
            temp_molar_fraction = in_fraction_value / mw
            temp_molar_fraction = temp_molar_fraction / temp_molar_fraction.sum(dim=-1, keepdim=True)
            temp_fraction = temp_molar_fraction * vm_298K
            out_fraction_value = temp_fraction / temp_fraction.sum(dim=-1, keepdim=True)
            return out_fraction_value

    elif in_fraction_type == 'volume':
        if out_fraction_type == 'molar':
            # Convert from volume fraction to molar fraction
            temp_fraction = in_fraction_value / vm_298K
            out_fraction_value = temp_fraction / temp_fraction.sum(dim=-1, keepdim=True)
            return out_fraction_value

        elif out_fraction_type == 'mass':
            # Convert from volume fraction to mass fraction
            temp_molar_fraction = in_fraction_value / vm_298K
            temp_molar_fraction = temp_molar_fraction / temp_molar_fraction.sum(dim=-1, keepdim=True)
            temp_fraction = temp_molar_fraction * mw
            out_fraction_value = temp_fraction / temp_fraction.sum(dim=-1, keepdim=True)
            return out_fraction_value

    return None  # Return None if no valid conversion found

def output_denormalize(y_minmax, true_values, predicted_values):
    # Fixme, should be improved later
    for ihead in range(len(y_minmax)):
        for isamp in range(len(predicted_values[0])):
            for iatom in range(len(predicted_values[ihead][0])):
                ymin = y_minmax[ihead][0][iatom]
                ymax = y_minmax[ihead][1][iatom]

                predicted_values[ihead][isamp][iatom] = (
                    predicted_values[ihead][isamp][iatom] * (ymax - ymin) + ymin
                )
                true_values[ihead][isamp][iatom] = (
                    true_values[ihead][isamp][iatom] * (ymax - ymin) + ymin
                )

    return true_values, predicted_values

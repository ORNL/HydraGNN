import torch


class LJpotential:

    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma

    def potential_energy(self, distance_vector):
        pair_distance = torch.norm(distance_vector)
        return (
            4
            * self.epsilon
            * ((self.sigma / pair_distance) ** 12 - (self.sigma / pair_distance) ** 6)
        )

    def radial_derivative(self, distance_vector):
        pair_distance = torch.norm(distance_vector)
        return (
            4
            * self.epsilon
            * (
                -12 * (self.sigma / pair_distance) ** 12 * 1 / pair_distance
                + 6 * (self.sigma / pair_distance) ** 6 * 1 / pair_distance
            )
        )

    def derivative_x(self, distance_vector):
        pair_distance = torch.norm(distance_vector)
        radial_derivative = self.radial_derivative(pair_distance)
        return radial_derivative * (distance_vector[0].item()) / pair_distance

    def derivative_y(self, distance_vector):
        pair_distance = torch.norm(distance_vector)
        radial_derivative = self.radial_derivative(pair_distance)
        return radial_derivative * (distance_vector[1].item()) / pair_distance

    def derivative_z(self, distance_vector):
        pair_distance = torch.norm(distance_vector)
        radial_derivative = self.radial_derivative(pair_distance)
        return radial_derivative * (distance_vector[2].item()) / pair_distance

import random
import sys
import tqdm
import numpy as np
import matplotlib.pyplot as plt

def gale_shapley(n, doctor_preferences, hospital_preferences):
    """
    Gale-Shapley algorithm for doctor hospital matching.

    Parameters:
    n (int): Number of doctors and hospitals.
    doctor_preferences (list of list of int): Preferences of doctors for hospitals.
    hospital_preferences (list of list of int): Preferences of hospitals for doctors.

    Returns:
    list of int: Matching of doctors to hospitals.
    """

    # Initialize free doctors and hospital preferences
    doctors = list(range(n))
    free_doctors = doctors[:]
    doctor_matches = [-1] * n
    hospital_matches = [-1] * n
    hospital_rankings = [None] * n
    for i in range(n):
        hospital_rankings[i] = {doctor: rank for rank, doctor in enumerate(hospital_preferences[i])}

    proposal_count = [0] * n  # Track how many proposals each doctor has made
    # While there are free doctors
    while free_doctors:
        # Pick the first free doctor
        doctor = free_doctors.pop(0)
        # Get current hospital preference
        hospital = doctor_preferences[doctor][proposal_count[doctor]]
        proposal_count[doctor] += 1
        # If the hospital is free, match them
        if hospital_matches[hospital] == -1:
            doctor_matches[doctor] = hospital
            hospital_matches[hospital] = doctor
        else:
            # If the hospital is not free, check if it prefers the new doctor
            current_doctor = hospital_matches[hospital]
            if hospital_rankings[hospital][doctor] < hospital_rankings[hospital][current_doctor]:
                # The hospital prefers the new doctor
                doctor_matches[doctor] = hospital
                doctor_matches[current_doctor] = -1
                free_doctors.append(current_doctor)
                hospital_matches[hospital] = doctor
            else:
                # The hospital prefers the current doctor
                doctor_matches[doctor] = -1
                free_doctors.append(doctor)
    return doctor_matches, proposal_count

def verify_matching(matching, doctor_preferences, hospital_preferences):
    """
    Verify the matching is stable.

    Parameters:
    matching (list of int): The matching of doctors to hospitals.
    doctor_preferences (list of list of int): Preferences of doctors for hospitals.
    hospital_preferences (list of list of int): Preferences of hospitals for doctors.

    Returns:
    bool: True if the matching is stable, False otherwise.
    """

    doctor_rankings = [{hospital: rank for rank, hospital in enumerate(doctor_preferences[i])} for i in range(len(matching))]
    hospital_rankings = [{doctor: rank for rank, doctor in enumerate(hospital_preferences[i])} for i in range(len(matching))]

    hospital_matching = [-1]  * len(matching)
    for doctor, hospital in enumerate(matching):
        hospital_matching[hospital] = doctor

    for d in range(len(matching)):
        current_hospital = matching[d]
        for h in range(len(matching)):
            if h == current_hospital:
                continue
            # Check if doctor d prefers hospital h over their current match
            if doctor_rankings[d][h] < doctor_rankings[d][current_hospital]:
                # Check if hospital h prefers doctor d over its current match
                current_doctor = hospital_matching[h]
                if current_doctor == -1 or hospital_rankings[h][d] < hospital_rankings[h][current_doctor]:
                    print(f"Unstable matching found: Doctor {d} prefers Hospital {h} over Hospital {current_hospital}, and Hospital {h} prefers Doctor {d} over Doctor {current_doctor}.")
                    return False
    return True

def validate_preferences(preferences):
    """
    Validate the preferences of doctors and hospitals.

    Parameters:
    preferences (list of list of int): Preferences of doctors for hospitals.

    Returns:
    bool: True if the preferences are valid, False otherwise.
    """
    n = len(preferences)
    for i in range(n):
        if len(preferences[i]) != n:
            print(f"Doctor {i} has invalid preferences.")
            return False
        if len(set(preferences[i])) != n:
            print(f"Doctor {i} has duplicate preferences.")
            return False
    return True

def random_preferences(n):
    """
    Generate random preferences for doctors and hospitals.

    Parameters:
    n (int): Number of doctors and hospitals.

    Returns:
    tuple: Two lists of preferences for doctors and hospitals.
    """
    doctors = list(range(n))
    hospitals = list(range(n))

    doctor_preferences = [random.sample(hospitals, n) for _ in range(n)]
    hospital_preferences = [random.sample(doctors, n) for _ in range(n)]

    return doctor_preferences, hospital_preferences

def lsm_preferences(n, lamb):
    """
    Generate preferences for doctors and hospitals based on a linear separable model

    Parameters:
    n (int): Number of doctors and hospitals.
    lamb (float): Lambda parameter for the linear separable model.

    Returns:
    tuple: Two lists of preferences for doctors and hospitals.
    """
    doctors = list(range(n))
    hospitals = list(range(n))
    for i in range(n):
        hospital_public_utility = np.random.uniform(0, 1)
        doctor_public_utility = np.random.uniform(0, 1)
        doctors[i] = doctor_public_utility
        hospitals[i] = hospital_public_utility

    doctor_preferences = []
    hospital_preferences = []
    for doctor in doctors:
        # Generate radom private utilities for the hospitals for the doctor
        private_utilities = np.random.uniform(0, 1, n)
        # Now generate the utilities for the hospitals according to this doctor
        utilities = np.zeros(n)
        for i, hospital_public in enumerate(hospitals):
            utilities[i] = lamb * hospital_public + (1 - lamb) * private_utilities[i]
        # Now sort the hospitals according to the utilities
        sorted_hospitals = np.argsort(-utilities)
        doctor_preferences.append(sorted_hospitals.tolist())

    for hospital in hospitals:
        # Generate random private utilities for the doctors for the hospital
        private_utilities = np.random.uniform(0, 1, n)
        # Now generate the utilities for the doctors according to this hospital
        utilities = np.zeros(n)
        for i, doctor_public in enumerate(doctors):
            utilities[i] = lamb * doctor_public + (1 - lamb) * private_utilities[i]
        # Now sort the doctors according to the utilities
        sorted_doctors = np.argsort(-utilities)
        hospital_preferences.append(sorted_doctors.tolist())
    return doctor_preferences, hospital_preferences

def calculate_rankings(matching, preferences):
    '''
    Calculate the rankings of the matching.
    Parameters:
    matching (list of int): The matching of doctors to hospitals (or vice versa).
    preferences (list of list of int): The preferences of doctors for hospitals (or vice versa).
    Returns:
    list of int: The rankings of the matching.
    '''
    rankings = [0] * len(matching)
    for i, match in enumerate(matching):
        rankings[i] = preferences[i].index(match) + 1
    return rankings

def main():
    mode = sys.argv[1]
    if mode not in ["random", "lsm"]:
        print("Please provide a valid mode: 'random' or 'lsm'.")
        return
    n = sys.argv[2]
    if n.isdigit():
        n = int(n)
    else:
        print("Please provide a valid integer for the number of doctors and hospitals.")
        return
    iterations = sys.argv[3]
    if iterations.isdigit():
        iterations = int(iterations)
    else:
        print("Please provide a valid integer for the number of iterations.")
        return
    n_bins = sys.argv[4]
    tick_spacing = sys.argv[5]
    if mode == "lsm":
        lamb = sys.argv[6]
        if lamb.replace('.', '', 1).isdigit():
            lamb = float(lamb)
            print(f"Using lambda: {lamb}")
        else:
            print("Please provide a valid float for the lambda parameter.")
            return

    avg_proposals = 0
    avg_doctor_rankings = 0
    avg_hospital_rankings = 0
    n_bins = int(n_bins)
    if n_bins < 1 or n_bins > n:
        print("Number of bins must be between 1 and n.")
        return
    bin_size = n // n_bins
    bin_counts_doctor = [0] * n_bins
    bin_counts_hospital = [0] * n_bins
    for i in tqdm.tqdm(range(iterations)):
        doctor_preferences, hospital_preferences = random_preferences(n) if mode == "random" else lsm_preferences(n, lamb)
        if not validate_preferences(doctor_preferences) or not validate_preferences(hospital_preferences):
            print("Invalid preferences generated.")
            return
        matching, proposal_count = gale_shapley(n, doctor_preferences, hospital_preferences)
        if not verify_matching(matching, doctor_preferences, hospital_preferences):
            print("Matching is not stable.")
            return
        doctor_rankings = calculate_rankings(matching, doctor_preferences)
        hospital_matching = [-1] * n
        for doctor, hospital in enumerate(matching):
            hospital_matching[hospital] = doctor
        hospital_rankings = calculate_rankings(hospital_matching, hospital_preferences)
        for doc_rank, hosp_rank in zip(doctor_rankings, hospital_rankings):
            bin_counts_doctor[(doc_rank - 1) // bin_size] += 1
            bin_counts_hospital[(hosp_rank - 1) // bin_size] += 1
        avg_proposals += sum(proposal_count) / n
        avg_doctor_rankings += sum(doctor_rankings) / n
        avg_hospital_rankings += sum(hospital_rankings) / n

    avg_proposals /= iterations
    avg_doctor_rankings /= iterations
    avg_hospital_rankings /= iterations
    bin_counts_doctor = np.array(bin_counts_doctor) / iterations
    bin_counts_hospital = np.array(bin_counts_hospital) / iterations
    print(f"Average number of proposals per doctor: {avg_proposals:.2f}")
    print(f"Average doctor rankings: {avg_doctor_rankings:.2f}")
    print(f"Average hospital rankings: {avg_hospital_rankings:.2f}")
    print(f"Expected doctor ranking based on log scaling: {np.log(n):.2f}")
    print(f"Expected hospital ranking based on log scaling: {n/np.log(n):.2f}")

    print("Doctor ranking distribution:")
    x_edges = np.arange(n_bins) * bin_size
    plt.bar(x_edges, bin_counts_doctor, color='steelblue', alpha=0.7, width= bin_size, align='edge', edgecolor='black')
    tick_spacing = int(tick_spacing)
    if tick_spacing < 1 or tick_spacing > n_bins:
        print("Tick spacing must be between 1 and n_bins.")
        return
    ticks = [f"{i+1}" for i in x_edges[::tick_spacing]]
    plt.xticks(x_edges[::tick_spacing], ticks, rotation=30)
    plt.xlabel('Ranking')
    plt.ylabel('Count')
    plt.title('Doctor Ranking Distribution')
    plt.legend()
    plt.show()

    print("Hospital ranking distribution:")
    plt.bar(x_edges, bin_counts_hospital, color='darkorange', alpha=0.7, width= bin_size, align='edge', edgecolor='black')
    plt.xticks(x_edges[::tick_spacing], ticks, rotation=30)
    plt.title('Hospital Ranking Distribution')
    plt.xlabel('Ranking')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    print("Histogram of proposals per doctor:")
    plt.hist(proposal_count, bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('Number of Proposals')
    plt.ylabel('Count')
    plt.title('Histogram of Proposals per Doctor')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt


def hysteresis_model(
    x, ascending_threshold, descending_threshold, low_value=0, high_value=1
):
    """
    A simple hysteresis function model.

    Parameters:
    -----------
    x : array-like
        Input signal
    ascending_threshold : float
        Threshold for switching from low to high state when input is increasing
    descending_threshold : float
        Threshold for switching from high to low state when input is decreasing
    low_value : float, optional
        Output value in the low state
    high_value : float, optional
        Output value in the high state

    Returns:
    --------
    array-like
        Output signal with hysteresis effect
    """
    if ascending_threshold <= descending_threshold:
        raise ValueError(
            "ascending_threshold must be greater than descending_threshold"
        )

    output = np.zeros_like(x)
    state = False  # Initial state (False = low, True = high)

    for i in range(len(x)):
        if state == False and x[i] > ascending_threshold:
            state = True
        elif state == True and x[i] < descending_threshold:
            state = False

        output[i] = high_value if state else low_value

    return output


def multi_level_hysteresis(x, thresholds, low_values, high_values):
    """
    A hysteresis function model with multiple threshold levels, each with its own low and high output values.

    Parameters:
    -----------
    x : array-like
        Input signal
    thresholds : list of tuples
        List of (ascending_threshold, descending_threshold) pairs for each level
        Must be ordered from lowest to highest threshold level
    low_values : list of float
        Values to output when transitioning from high to low state at each threshold
    high_values : list of float
        Values to output when transitioning from low to high state at each threshold

    Returns:
    --------
    array-like
        Output signal with multi-level hysteresis effect
    """
    if len(thresholds) != len(low_values) or len(thresholds) != len(high_values):
        raise ValueError(
            "thresholds, low_values, and high_values must have the same length"
        )

    # Extract ascending and descending thresholds
    ascending_thresholds = [t[0] for t in thresholds]
    descending_thresholds = [t[1] for t in thresholds]

    # Check that ascending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(ascending_thresholds[:-1], ascending_thresholds[1:])
    ):
        raise ValueError("ascending thresholds must be in ascending order")

    # Check that descending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(descending_thresholds[:-1], descending_thresholds[1:])
    ):
        raise ValueError("descending thresholds must be in ascending order")

    # Check that each descending threshold is less than its corresponding ascending threshold
    if not all(d < a for d, a in zip(descending_thresholds, ascending_thresholds)):
        raise ValueError(
            "Each descending threshold must be less than its corresponding ascending threshold"
        )

    output = np.zeros_like(x)

    # CHANGE: Track active threshold level instead of binary state
    active_level = -1  # Start with no active level

    # For the first point, determine initial level based on value
    for i in range(len(thresholds)):
        if x[0] > ascending_thresholds[i]:
            active_level = i

    # Apply initial output value
    if active_level >= 0:
        output[0] = high_values[active_level]
    else:
        output[0] = low_values[0] if low_values else 0

    # Process the rest of the points
    for i in range(1, len(x)):
        current_value = x[i]
        previous_level = active_level

        # MAJOR CHANGE: Complete rewrite of level transition logic
        # Check if we need to change level
        if active_level >= 0:
            # We're already in a high state at some level
            # Check if we should go up to a higher level
            for j in range(active_level + 1, len(thresholds)):
                if current_value > ascending_thresholds[j]:
                    active_level = j

            # Check if we should go down (potentially multiple levels)
            if current_value < descending_thresholds[active_level]:
                # Find the highest level that's still active
                new_level = -1
                for j in range(active_level):
                    if current_value > descending_thresholds[j]:
                        new_level = j
                active_level = new_level
        else:
            # We're in the lowest state, check if we should go up
            for j in range(len(thresholds)):
                if current_value > ascending_thresholds[j]:
                    active_level = j

        # Set output based on level change
        if active_level != previous_level:
            if active_level >= 0:
                output[i] = high_values[active_level]
            else:
                output[i] = low_values[0] if low_values else 0
        else:
            # No level change, keep previous output
            output[i] = output[i - 1]

    return output


def old_multi_level_hysteresis(x, thresholds, low_values, high_values):
    """
    A hysteresis function model with multiple threshold levels, each with its own low and high output values.

    Parameters:
    -----------
    x : array-like
        Input signal
    thresholds : list of tuples
        List of (ascending_threshold, descending_threshold) pairs for each level
        Must be ordered from lowest to highest threshold level
    low_values : list of float
        Values to output when transitioning from high to low state at each threshold
    high_values : list of float
        Values to output when transitioning from low to high state at each threshold

    Returns:
    --------
    array-like
        Output signal with multi-level hysteresis effect
    """
    if len(thresholds) != len(low_values) or len(thresholds) != len(high_values):
        raise ValueError(
            "thresholds, low_values, and high_values must have the same length"
        )

    # Extract ascending and descending thresholds
    ascending_thresholds = [t[0] for t in thresholds]
    descending_thresholds = [t[1] for t in thresholds]

    # Check that ascending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(ascending_thresholds[:-1], ascending_thresholds[1:])
    ):
        raise ValueError("ascending thresholds must be in ascending order")

    # Check that descending thresholds are in ascending order
    if not all(
        a < b for a, b in zip(descending_thresholds[:-1], descending_thresholds[1:])
    ):
        raise ValueError("descending thresholds must be in ascending order")

    # Check that each descending threshold is less than its corresponding ascending threshold
    if not all(d < a for d, a in zip(descending_thresholds, ascending_thresholds)):
        raise ValueError(
            "Each descending threshold must be less than its corresponding ascending threshold"
        )

    output = np.zeros_like(x)
    state = np.zeros(
        len(x), dtype=bool
    )  # Track state for each point (False = low, True = high)

    # For the first point, determine initial state based on value
    # If starting value is already above any ascending threshold, set initial state accordingly
    for i in range(len(thresholds)):
        if x[0] > ascending_thresholds[i]:
            state[0] = True

    # Apply the appropriate initial value
    if state[0]:
        # Find the highest threshold level that the initial value has crossed
        level = max(
            [i for i in range(len(thresholds)) if x[0] > ascending_thresholds[i]],
            default=0,
        )
        output[0] = high_values[level]
    else:
        # Find the highest threshold level where the initial value is above the descending threshold
        level = max(
            [i for i in range(len(thresholds)) if x[0] > descending_thresholds[i]],
            default=-1,
        )
        if level >= 0:
            output[0] = low_values[level]
        else:
            output[0] = low_values[0]  # If below all thresholds, use the lowest value

    print(f"init: x[0]={x[0]}, state[0]={state[0]} -> {output[0]}")

    # Process the rest of the points
    for i in range(1, len(x)):
        # Start with previous state
        state[i] = state[i - 1]
        print(f"x[{i}]={x[i]}, init_state[{i}]={state[i]}", end="", flush=True)

        # Assume No state change, maintain previous output
        output[i] = output[i - 1]

        # Check each threshold level
        for j in range(len(thresholds)):
            if not state[i] and x[i] >= ascending_thresholds[j]:
                # Transition from low to high
                state[i] = True
                output[i] = high_values[j]
                print("(* ", j, ascending_thresholds[j], end=")", flush=True)
                break
            elif state[i] and x[i] <= descending_thresholds[j]:
                # Transition from high to low
                state[i] = False
                output[i] = low_values[j]
                print("(x ", j, descending_thresholds[j], end=")", flush=True)
                break
        print(f" new_state[{i}]={state[i]} -> {output[i]}", flush=True)

    print(f"multi: , output={type(output)}")
    return output


def relay_hysteresis(x, center, width, low_value=0, high_value=1):
    """
    A relay-type hysteresis function centered around a specific value.

    Parameters:
    -----------
    x : array-like
        Input signal
    center : float
        Center point of the hysteresis loop
    width : float
        Width of the hysteresis loop
    low_value : float, optional
        Output value in the low state
    high_value : float, optional
        Output value in the high state

    Returns:
    --------
    array-like
        Output signal with hysteresis effect
    """
    ascending_threshold = center + width / 2
    descending_threshold = center - width / 2

    return hysteresis_model(
        x, ascending_threshold, descending_threshold, low_value, high_value
    )


def continuous_hysteresis(x, center, width, slope=10, low_value=0, high_value=1):
    """
    A continuous hysteresis function with smooth transitions.

    Parameters:
    -----------
    x : array-like
        Input signal
    center : float
        Center point of the hysteresis loop
    width : float
        Width of the hysteresis loop
    slope : float, optional
        Controls the steepness of the transitions
    low_value : float, optional
        Minimum output value
    high_value : float, optional
        Maximum output value

    Returns:
    --------
    array-like
        Output signal with continuous hysteresis effect
    """
    output = np.zeros_like(x)
    state = np.zeros_like(x)

    # Initial state
    state[0] = 0.5

    # Calculate state using a differential equation that exhibits hysteresis
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]

        if dx > 0:  # Ascending
            target = 1 / (1 + np.exp(-slope * (x[i] - (center + width / 2))))
        else:  # Descending or constant
            target = 1 / (1 + np.exp(-slope * (x[i] - (center - width / 2))))

        # Simple relaxation to target
        state[i] = state[i - 1] + 0.1 * (target - state[i - 1])

    # Scale to desired output range
    output = low_value + state * (high_value - low_value)

    return output


# Example usage
if __name__ == "__main__":
    # Generate a sine wave input
    t = np.linspace(0, 4 * np.pi, 1000)
    x = np.sin(t)

    # Model different types of hysteresis
    y1 = hysteresis_model(x, 0.5, -0.5)
    y2 = relay_hysteresis(x, 0, 1)
    y3 = continuous_hysteresis(x, 0, 1, slope=8)

    # Plotting
    plt.figure(figsize=(15, 10))

    # Plot input signal
    plt.subplot(2, 2, 1)
    plt.plot(t, x)
    plt.title("Input Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Plot different hysteresis functions
    plt.subplot(2, 2, 2)
    plt.plot(x, y1)
    plt.title("Discrete Hysteresis")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x, y2)
    plt.title("Relay Hysteresis")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x, y3)
    plt.title("Continuous Hysteresis")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot time-domain responses
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(t, x)
    plt.title("Input Signal vs Time")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t, y1)
    plt.title("Discrete Hysteresis vs Time")
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(t, y2)
    plt.title("Relay Hysteresis vs Time")
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(t, y3)
    plt.title("Continuous Hysteresis vs Time")
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Pole-Zero plot function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def zplane(b, a=None, ax=None):
    """
    Plot pole-zero diagram for a discrete-time system.
    
    Parameters:
    - b (array-like): Numerator polynomial coefficients or zeros
    - a (array-like, optional): Denominator polynomial coefficients or poles
    - ax (matplotlib axis, optional): Axis to plot on. If None, creates new figure.
    
    Returns:
    - ax: The matplotlib axis object
    
    Example:
        zplane([1, -1.96, 0.97])  # Plot poles of polynomial
        zplane([1], [1, -1.96, 0.97])  # Plot zeros and poles
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Get zeros and poles
    if a is None:
        # Only one argument: treat as poles (denominator)
        if len(b) > 1:
            poles = np.roots(b)
        else:
            poles = np.array([])
        zeros = np.array([])
    else:
        # Two arguments: numerator and denominator
        if len(b) > 1:
            zeros = np.roots(b)
        else:
            zeros = np.array([])
        if len(a) > 1:
            poles = np.roots(a)
        else:
            poles = np.array([])
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)
    
    # Plot zeros (circles)
    if len(zeros) > 0:
        ax.plot(np.real(zeros), np.imag(zeros), 'o', markersize=8, 
                markerfacecolor='none', markeredgewidth=1.5, markeredgecolor='b')
    
    # Plot poles (x markers)
    if len(poles) > 0:
        ax.plot(np.real(poles), np.imag(poles), 'x', markersize=10, 
                markeredgewidth=2, markeredgecolor='r')
    
    # Add axes through origin
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Set labels and formatting
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_aspect('equal', adjustable='box')
    
    # Set axis limits to show unit circle nicely
    max_val = 1.5
    if len(poles) > 0:
        max_val = max(1.5, 1.2 * max(np.max(np.abs(np.real(poles))), np.max(np.abs(np.imag(poles)))))
    if len(zeros) > 0:
        max_val = max(max_val, 1.2 * max(np.max(np.abs(np.real(zeros))), np.max(np.abs(np.imag(zeros)))))
    
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    
    return ax

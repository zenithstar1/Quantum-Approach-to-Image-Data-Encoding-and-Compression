# Import necessary libraries
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path, size=(64, 64)):
    """Load an image, resize it, and convert it to a numpy array."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size)  # Increase resolution to 64x64 pixels
        print(f"Image loaded successfully: {image_path}")
        return np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def amplitude_encode(image_array):
    """Amplitude encode the image data into a quantum circuit."""
    flat_img = image_array.flatten().astype(np.float32)
    norm_img = flat_img / np.linalg.norm(flat_img)

    num_amplitudes = len(norm_img)
    num_qubits = int(np.ceil(np.log2(num_amplitudes)))

    padded_img = np.pad(norm_img, (0, 2**num_qubits - num_amplitudes))

    circuit = QuantumCircuit(num_qubits)

    for i in range(2**num_qubits):
        theta = 2 * np.arccos(padded_img[i]) if padded_img[i] > 0 else 0
        circuit.ry(theta, i % num_qubits)

    print(f"Amplitude encoding completed for {len(padded_img)} amplitudes using {num_qubits} qubits.")
    return circuit, padded_img

def quantum_compression_technique_1(encoded_amplitudes):
    """Apply improved Quantum Compression Technique 1 (QCT1) to the encoded amplitudes."""
    num_qubits = int(np.ceil(np.log2(len(encoded_amplitudes))))
    circuit = QuantumCircuit(num_qubits)

    # Apply more sophisticated quantum gates for improved compression
    for i in range(num_qubits - 1):
        circuit.h(i)
        circuit.cx(i, i + 1)
        circuit.z(i + 1)  # Replace RZ gate with Pauli Z gate
        circuit.cx(i, i + 1)
        circuit.h(i)

    simulator = Aer.get_backend('aer_simulator')
    transpiled_circuit = transpile(circuit, simulator)
    job = simulator.run(transpiled_circuit)
    result = job.result()

    # Implement a more gradual compression technique
    compression_factor = 0.75 # Adjust this value to control compression level
    compressed_length = int(len(encoded_amplitudes) * compression_factor)
    compressed_amplitudes = encoded_amplitudes[:compressed_length]
    compressed_amplitudes /= np.linalg.norm(compressed_amplitudes)

    print(f"Compression completed. Compressed amplitudes reduced to {len(compressed_amplitudes)}.")
    return compressed_amplitudes

def visualize_images(original_image, encoded_amplitudes, compressed_amplitudes):
    """Visualize the original, encoded, and compressed images side by side."""
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    

    compressed_image = reconstruct_image(compressed_amplitudes, original_image.shape)
    plt.subplot(1, 2, 2)
    plt.imshow(compressed_image)
    plt.title('Compressed Image (QCT1)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('quantum_image_processing_results.png', dpi=300)  # Save high-resolution image
    plt.show()

def reconstruct_image(amplitudes, original_shape):
    """Reconstruct an image from given amplitudes with improved quality."""
    num_pixels = np.prod(original_shape)

    if len(amplitudes) > num_pixels:
        truncated_amplitudes = amplitudes[:num_pixels]
    else:
        truncated_amplitudes = np.pad(amplitudes, (0, num_pixels - len(amplitudes)))

    reconstructed = truncated_amplitudes.reshape(original_shape)

    # Apply contrast stretching for better visibility
    p2, p98 = np.percentile(reconstructed, (2, 98))
    reconstructed = np.clip(reconstructed, p2, p98)
    
    reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min()) * 255
    return reconstructed.astype(np.uint8)

if __name__ == "__main__":
    img_path = r"C:\Users\anuja\OneDrive\文档\Quantum\venv\WhatsApp Image 2024-10-13 at 6.49.45 PM.jpeg"  # Update with your image path
    img_array = load_image(img_path, size=(64, 64))  # Increased resolution

    if img_array is not None:
        circuit, encoded_amplitudes = amplitude_encode(img_array)
        compressed_amplitudes = quantum_compression_technique_1(encoded_amplitudes)
        visualize_images(img_array, encoded_amplitudes, compressed_amplitudes)
    else:
        print("Failed to load image, exiting the program.")

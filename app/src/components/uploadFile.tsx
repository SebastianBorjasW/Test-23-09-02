import { useState, useEffect, ChangeEvent } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { API_URL } from "../auth/constants";

export default function FileUploader() {
    const [selectedImages, setSelectedImages] = useState<File[]>([]);

    useEffect(() => {
        const preventDefault = (e: Event) => e.preventDefault();

        window.addEventListener("dragover", preventDefault);
        window.addEventListener("drop", preventDefault);

        return () => {
            window.removeEventListener("dragover", preventDefault);
            window.removeEventListener("drop", preventDefault);
        };
    }, []);

    // Manejo del drop de archivos
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop: (acceptedFiles) => {
            if (acceptedFiles.length > 0) {
                // Reemplazar imágenes anteriores con las nuevas
                setSelectedImages(acceptedFiles);
            }
        },
        accept: "image/*",
        multiple: true,  // Permitir múltiples imágenes
    });

    // Función para enviar las imágenes al servidor
    const handleSubmit = async (event: ChangeEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (selectedImages.length === 0) {
            alert("Por favor selecciona al menos una imagen");
            return;
        }

        const formData = new FormData();
        selectedImages.forEach((image) => {
            formData.append("files", image);  // Cambiar "file" a "files" para aceptar múltiples imágenes
        });

        try {
            const response = await axios.post(`${API_URL}/load_img`, formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            if (response.status === 200) {
                alert("Imágenes subidas con éxito");
            } else {
                alert("Error al subir las imágenes");
            }
        } catch (error) {
            console.error("Error al enviar las imágenes:", error);
            alert("Ocurrió un error al intentar subir las imágenes.");
        }
    };

    return (
        <div className="min-h-screen flex justify-center items-center">
            <form onSubmit={handleSubmit} className="flex flex-col items-center">
                <div
                    {...getRootProps({
                        className:
                            "border-dashed border-2 border-gray-400 p-6 rounded-lg w-96 text-center cursor-pointer",
                    })}
                >
                    <input {...getInputProps()} type="file" accept="image/*" multiple />  {/* Cambiado para aceptar múltiples imágenes */}
                    {isDragActive ? (
                        <p className="text-gray-500">Drop images here...</p>
                    ) : (
                        <div>
                            <img
                                src="../../public/icons/cloud-upload.svg"
                                alt="Upload"
                                className="mx-auto mb-4 w-16 h-16"
                            />
                            <p className="text-gray-500 text-xl">Drag&Drop images here</p>
                            <p className="text-gray-500">or</p>
                            <button
                                type="button"
                                className="bg-white text-cyan-500 border-2 border-cyan-500 px-4 py-2 rounded-lg mt-2"
                            >
                                Browse Images
                            </button>
                        </div>
                    )}
                </div>

                {/* Mostrar vista previa solo de la primera imagen seleccionada */}
                {selectedImages.length > 0 && (
                    <div className="mt-4">
                        <p className="text-gray-500">Selected image: {selectedImages[0].name}</p>
                        <img
                            src={URL.createObjectURL(selectedImages[0])}
                            alt="Vista previa"
                            className="h-48 w-48 object-cover mt-4"
                        />
                    </div>
                )}

                <button
                    type="submit"
                    className="bg-white text-cyan-500 border-2 border-cyan-500 px-4 py-2 rounded-lg mt-4"
                >
                    Upload Images
                </button>
            </form>
        </div>
    );
}

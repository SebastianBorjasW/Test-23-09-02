import React, { useState, useEffect, ChangeEvent } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { API_URL } from "../auth/constants";

export default function FileUploader() {
    const [selectedImage, setSelectedImage] = useState<File | null>(null);

    // Prevenir comportamiento predeterminado en escritorio para drag & drop
    useEffect(() => {
        // Evita el comportamiento predeterminado del navegador
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
                setSelectedImage(acceptedFiles[0]);
            }
        },
        accept: "image/*",
        multiple: false,
    });

    // Función para enviar la imagen al servidor
    const handleSubmit = async (event: ChangeEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (!selectedImage) {
            alert("Por favor selecciona una imagen");
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedImage);

        try {
            const response = await axios.post(`${API_URL}/load_img`, formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            if (response.status === 200) {
                alert("Imagen subida con éxito");
            } else {
                alert("Error al subir la imagen");
            }
        } catch (error) {
            console.error("Error al enviar la imagen:", error);
            alert("Ocurrió un error al intentar subir la imagen.");
        }
    };

    return (
        <div className="bg-cyan-900 min-h-screen flex justify-center items-center">
            <form onSubmit={handleSubmit} className="flex flex-col items-center">
                <div
                    {...getRootProps({
                        className:
                            "border-dashed border-2 border-gray-400 p-6 rounded-lg w-96 text-center cursor-pointer",
                    })}
                >
                    <input {...getInputProps()} 
                    type="file"
                    accept="image/*"
                     />
                    {isDragActive ? (
                        <p className="text-gray-500">Suelta los archivos aquí...</p>
                    ) : (
                        <div>
                            <img
                                src="/upload-icon.png"
                                alt="Upload Icon"
                                className="mx-auto mb-4"
                            />
                            <p className="text-gray-500">Arrastra y suelta archivos aquí</p>
                            <p className="text-gray-500">o</p>
                            <button
                                type="button"
                                className="bg-blue-500 text-white px-4 py-2 rounded-lg mt-2"
                            >
                                Buscar archivos
                            </button>
                        </div>
                    )}
                </div>

                {selectedImage && (
                    <div className="mt-4">
                        <p className="text-white">Archivo seleccionado: {selectedImage.name}</p>
                        <img
                            src={URL.createObjectURL(selectedImage)}
                            alt="Vista previa"
                            className="h-48 w-48 object-cover mt-4"
                        />
                    </div>
                )}

                <button
                    type="submit"
                    className="bg-blue-500 text-white px-4 py-2 rounded-lg mt-4"
                >
                    Subir Imagen
                </button>
            </form>
        </div>
    );
}

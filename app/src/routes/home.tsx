import React, { useState, ChangeEvent, FormEvent } from "react";
import { API_URL } from "../auth/constants";
import axios from 'axios';


export default function Home() {
    // Estado para almacenar la imagen seleccionada
    const [selectedImage, setSelectedImage] = useState<File | null>(null);

    // Función para manejar la selección de una imagen
    const handleImageChange = (event: ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files;
        if (files && files.length > 0) {
            setSelectedImage(files[0]); // Guardamos el archivo seleccionado en el estado
        }
    };

    // Función para enviar la imagen al servidor
    const handleSubmit = async (event: FormEvent) => {
        event.preventDefault(); // Prevenir que se recargue la página

        if (!selectedImage) {
            alert("Por favor selecciona una imagen");
            return;
        }

        // Crear un objeto FormData para enviar el archivo
        const formData = new FormData();
        formData.append("file", selectedImage);

        try {
            // Realizar la petición HTTP con Axios para enviar la imagen al servidor
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
        <div className="bg-cyan-900 min-h-screen">
            <div>
                <h1 className="text-3xl font-bold underline flex justify-center items-center py-5">
                    Clasificador de frutas
                </h1>
            </div>

            {/* Formulario para seleccionar y enviar la imagen */}
            <form onSubmit={handleSubmit} className="flex flex-col items-center py-5">
                <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageChange}
                    className="mb-4"
                />
                <button
                    type="submit"
                    className="bg-blue-500 text-white px-4 py-2 rounded-lg"
                >
                    Subir Imagen
                </button>
            </form>
        </div>
    );
}

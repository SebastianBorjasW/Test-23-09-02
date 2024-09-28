import { useState, useEffect, ChangeEvent } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { API_URL } from "../auth/constants";

export default function FileUploader() {
    const [selectedImages, setSelectedImages] = useState<File[]>([]);
    const [uploadSuccess, setUploadSuccess] = useState(false); 
    const [downloadLink, setDownloadLink] = useState<string | null>(null); 

    useEffect(() => {
        const preventDefault = (e: Event) => e.preventDefault();

        window.addEventListener("dragover", preventDefault);
        window.addEventListener("drop", preventDefault);

        return () => {
            window.removeEventListener("dragover", preventDefault);
            window.removeEventListener("drop", preventDefault);
        };
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop: (acceptedFiles) => {
            if (acceptedFiles.length > 0) {
                setSelectedImages(acceptedFiles);
                setUploadSuccess(false);  
            }
        },
        accept: "image/*",
        multiple: true,  
    });

    const handleSubmit = async (event: ChangeEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (selectedImages.length === 0) {
            alert("Por favor selecciona al menos una imagen");
            return;
        }

        const formData = new FormData();
        selectedImages.forEach((image) => {
            formData.append("files", image); 
        });

        try {
            const response = await axios.post(`${API_URL}/load_img`, formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            if (response.status === 200) {
                setUploadSuccess(true);  
                alert("Imágenes subidas con éxito");

                const downloadResponse = await axios.get(`${API_URL}/load_img/download`, {
                    responseType: 'blob',  
                });

                const url = window.URL.createObjectURL(new Blob([downloadResponse.data]));
                setDownloadLink(url);  
            } else {
                alert("Error al subir las imágenes");
            }
        } catch (error) {
            console.error("Error al enviar las imágenes:", error);
            alert("Ocurrió un error al intentar subir las imágenes.");
        }
    };

    const getDisplayedFileNames = () => {
        if (selectedImages.length <= 2) {
            return selectedImages.map((file) => file.name).join(", ");
        } else {
            const firstFourFiles = selectedImages.slice(0, 2).map((file) => file.name);
            return `${firstFourFiles.join(", ")} ...`;
        }
    };

    return (
        <div className="min-h-screen flex justify-center items-center p-4">
            <form onSubmit={handleSubmit} className="flex flex-col items-center w-full max-w-xl">
                {/* Cuadro de drag & drop */}
                <div
                    {...getRootProps({
                        className:
                            "border-dashed border-2 border-gray-400 p-6 rounded-lg w-full text-center cursor-pointer sm:w-96",
                    })}
                >
                    <input {...getInputProps()} type="file" accept="image/*" multiple /> 
                    {isDragActive ? (
                        <p className="text-gray-500">Drop images here...</p>
                    ) : (
                        <div>
                            <img
                                src="../../public/icons/cloud-upload.svg"
                                alt="Upload"
                                className="mx-auto mb-4 w-12 h-12 sm:w-16 sm:h-16"
                            />
                            <p className="text-gray-500 text-lg sm:text-xl">Drag & Drop images here</p>
                            <p className="text-gray-500">or</p>
                            <button
                                type="button"
                                className="bg-white text-cyan-500 border-2 border-cyan-500 px-4 py-2 rounded-lg mt-2 shadow-lg"
                            >
                                Browse Images
                            </button>
                        </div>
                    )}
                </div>

                {/* Vista previa de las imágenes seleccionadas */}
                {!uploadSuccess && selectedImages.length > 0 && (
                    <div className="mt-4">
                        <p className="text-gray-500">Selected images: {getDisplayedFileNames()}</p>
                        <img
                            src={URL.createObjectURL(selectedImages[0])}
                            alt="Vista previa"
                            className="h-32 w-32 sm:h-48 sm:w-48 object-cover mt-4 mx-auto"
                        />
                    </div>
                )}

                {!uploadSuccess && (
                    <button
                        type="submit"
                        className="bg-white text-cyan-500 border-2 border-cyan-500 px-4 py-2 rounded-lg mt-4 shadow-lg"
                    >
                        Upload Images
                    </button>
                )}

                {/* Botón de descarga si la subida fue exitosa */}
                {uploadSuccess && downloadLink && (
                    <div className="mt-4">
                        <a
                            href={downloadLink}
                            download="classified_images.zip"
                            className="bg-white text-cyan-500 border-2 border-cyan-500 px-4 py-2 rounded-lg mt-4 shadow-lg"
                        >
                            Download Classified Images
                        </a>
                    </div>
                )}
            </form>
        </div>
    );
}

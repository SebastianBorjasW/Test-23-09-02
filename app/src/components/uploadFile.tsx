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
        if (selectedImages.length <= 3) {
            return selectedImages.map((file) => file.name).join(", ");
        } else {
            const firstFourFiles = selectedImages.slice(0, 3).map((file) => file.name);
            return `${firstFourFiles.join(", ")} ...`;
        }
    };

    return (
        <div className="min-h-screen flex justify-center items-center">
            <form onSubmit={handleSubmit} className="flex flex-col items-center">
                {/* El cuadro de drag & drop siempre debe estar visible */}
                <div
                    {...getRootProps({
                        className:
                            "border-dashed border-2 border-gray-400 p-6 rounded-lg w-96 text-center cursor-pointer",
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

                {/* Mostrar vista previa solo si las imágenes no han sido subidas */}
                {!uploadSuccess && selectedImages.length > 0 && (
                    <div className="mt-4">
                        <p className="text-gray-500">Selected images: {getDisplayedFileNames()}</p>
                        <img
                            src={URL.createObjectURL(selectedImages[0])}
                            alt="Vista previa"
                            className="h-48 w-48 object-cover mt-4 mx-auto"
                        />
                    </div>
                )}

                {!uploadSuccess && (
                    <button
                        type="submit"
                        className="bg-white text-cyan-500 border-2 border-cyan-500 px-4 py-2 rounded-lg mt-4"
                    >
                        Upload Images
                    </button>
                )}

                {/* Mostrar el botón de descarga solo si la subida fue exitosa */}
                {uploadSuccess && downloadLink && (
                    <div className="mt-4">
                        <a
                            href={downloadLink}
                            download="classified_images.zip"
                            className="bg-cyan-500 text-white px-4 py-2 rounded-lg"
                        >
                            Download Classified Images
                        </a>
                    </div>
                )}
            </form>
        </div>
    );
}

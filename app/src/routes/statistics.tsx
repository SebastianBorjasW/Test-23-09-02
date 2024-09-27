import ImagesShow from "../components/images_show";
import Sidebar from "../components/sideBar";

export default function FileUploader() {
    
    return (
        <div className="bg-white min-h-screen flex">
            {/* Barra lateral */}
            <Sidebar />

            {/* Contenido principal */}
            <div className="flex-grow flex justify-center items-center">
                <ImagesShow />
            </div>
        </div>
    );
}


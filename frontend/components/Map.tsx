import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { Ref, useCallback, useEffect, useMemo, useState } from "react";

const markerIcon = L.icon({
  iconUrl: "favicon.ico",
});

interface Pos {
  lat: number;
  lng: number;
  zoom: number;
}

const PositionComponent = ({ map }: { map: L.Map }) => {
  const [position, setPosition] = useState<Pos>({
    lat: map.getCenter().lat,
    lng: map.getCenter().lng,
    zoom: map.getZoom(),
  });
  const onChange = useCallback(() => {
    const latlng = map.getCenter();
    setPosition({
      lat: latlng.lat,
      lng: latlng.lng,
      zoom: map.getZoom(),
    });
  }, [map]);

  useEffect(() => {
    //listen for changes in movement or zooming
    map.on("move", onChange);
    map.on("zoom", onChange);
    return () => {
      map.off("move", onChange);
      map.off("zoom", onChange);
    };
  }, [map]);
  return (
    <p>{`Lat: ${position.lat.toFixed(4)}, Lng: ${position.lng.toFixed(
      4
    )}, zoom: ${position.zoom}`}</p>
  );
};

const Map = () => {
  const [map, setMap] = useState(null);

  const displayMap = useMemo(
    () => (
      <MapContainer
        center={[12.9846215, 77.6955662]}
        zoom={13}
        scrollWheelZoom={true}
        attributionControl={false}
        style={{ height: 400, width: "100%" }}
        //@ts-ignore
        ref={setMap}
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <Marker position={[51.505, -0.09]} icon={markerIcon}>
          <Popup>
            A pretty CSS3 popup. <br /> Easily customizable.
          </Popup>
        </Marker>
      </MapContainer>
    ),
    []
  );

  return (
    <div className="p-4">
      {displayMap}
      <p>{map ? <PositionComponent map={map} /> : "loading..."}</p>
    </div>
  );
};

export default Map;

import type { NextPage } from "next";
import Head from "next/head";
import dynamic from "next/dynamic";

const Map = dynamic(() => import("../components/Map"), {
  loading: () => <p>The map is loading</p>,
  ssr: false, // This line is important. It's what prevents server-side render
});
const Home: NextPage = () => {
  return (
    <div>
      <Head>
        <title>mapviz</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="text-center">
        <h1 className="text-2xl p-2">Map time 🗺️</h1>
        <Map />
      </main>
    </div>
  );
};

export default Home;

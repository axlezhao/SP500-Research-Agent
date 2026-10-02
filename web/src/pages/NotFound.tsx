import { Link } from "react-router";

export default function NotFound() {
  return (
    <div className="card mx-auto mt-16 max-w-md p-8 text-center">
      <h1 className="text-lg font-semibold">Page not found</h1>
      <p className="mt-1 text-[13px] text-text-2">That page doesn't exist.</p>
      <Link to="/" className="mt-4 inline-block rounded-lg bg-accent px-3 py-1.5 text-[13px] font-medium text-white">
        Back to the overview
      </Link>
    </div>
  );
}
